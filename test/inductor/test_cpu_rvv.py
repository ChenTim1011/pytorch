# Owner(s): ["module: inductor"]
import importlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch._inductor import cpu_vec_isa
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import HardwareClassification, run_tests


# Initialize lowering before importing templates that depend on kernel.mm.
importlib.import_module("torch._inductor.lowering")


class TestRvvMicroGemm(TestCase):
    hw_classification = HardwareClassification.CPU

    def test_cpp_gemm_cache_size_parser(self):
        from torch._inductor.codegen.cpp_gemm_template import (
            _parse_sysfs_cache_size,
            _read_riscv_sysfs_cache_sizes,
        )

        self.assertEqual(_parse_sysfs_cache_size("64K\n"), 64 * 1024)
        self.assertEqual(_parse_sysfs_cache_size("2M"), 2 * 1024 * 1024)
        self.assertEqual(_parse_sysfs_cache_size("4096"), 4096)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cpu_root = Path(tmp_dir)

            def write_cache(
                cpu: int, index: int, level: int, kind: str, size: str
            ) -> None:
                cache_dir = cpu_root / f"cpu{cpu}" / "cache" / f"index{index}"
                cache_dir.mkdir(parents=True)
                (cache_dir / "level").write_text(str(level))
                (cache_dir / "type").write_text(kind)
                (cache_dir / "size").write_text(size)

            write_cache(0, 0, 1, "Data", "64K")
            write_cache(0, 1, 2, "Unified", "2M")
            write_cache(1, 0, 1, "Data", "32K")
            write_cache(1, 1, 2, "Unified", "1M")
            write_cache(1, 2, 1, "Instruction", "16K")
            self.assertEqual(
                _read_riscv_sysfs_cache_sizes(cpu_root),
                (32 * 1024, 1024 * 1024),
            )

    def test_rvv_bf16_micro_gemm_registration(self):
        from torch._inductor.codegen.cpp_micro_gemm import (
            CppMicroGemmRVVBF16M1,
            CppMicroGemmRVVBF16MGe2,
            create_micro_gemm,
            micro_gemm_configs,
        )
        from torch._inductor.virtualized import V

        m1_config = micro_gemm_configs[CppMicroGemmRVVBF16M1][0]
        prefill_config = micro_gemm_configs[CppMicroGemmRVVBF16MGe2][0]
        self.assertIs(m1_config.vec_isa_cls, cpu_vec_isa.VecRVV)
        self.assertIs(prefill_config.vec_isa_cls, cpu_vec_isa.VecRVV)
        self.assertEqual(m1_config.register_blocking, (1, 32, 1))
        self.assertEqual(prefill_config.register_blocking, (4, 32, 1))
        self.assertTrue(
            m1_config.extra_check(m1_config, 1, 64, 64, 1, 1, dynamic_M=False)
        )
        self.assertFalse(
            m1_config.extra_check(m1_config, 2, 64, 64, 1, 1, dynamic_M=False)
        )
        self.assertTrue(
            prefill_config.extra_check(prefill_config, 2, 64, 64, 1, 1, dynamic_M=False)
        )
        self.assertFalse(
            prefill_config.extra_check(prefill_config, 2, 64, 64, 1, 1, dynamic_M=True)
        )

        class FakeSizeVars:
            @staticmethod
            def optimization_hint(value, fallback):
                return value

        class FakeGraph:
            sizevars = FakeSizeVars()

        class FakeKernel:
            assert_function = "TORCH_CHECK"

        representative_linear_shapes = [
            ("prefill_qkv", 64, 3072, 2048, CppMicroGemmRVVBF16MGe2, (4, 32, 1)),
            ("prefill_o_proj", 64, 2048, 2048, CppMicroGemmRVVBF16MGe2, (4, 32, 1)),
            ("prefill_gate_up", 64, 16384, 2048, CppMicroGemmRVVBF16MGe2, (4, 32, 1)),
            ("prefill_down", 64, 2048, 8192, CppMicroGemmRVVBF16MGe2, (4, 32, 1)),
            ("decode_qkv", 1, 3072, 2048, CppMicroGemmRVVBF16M1, (1, 32, 1)),
            ("decode_o_proj", 1, 2048, 2048, CppMicroGemmRVVBF16M1, (1, 32, 1)),
            ("decode_gate_up", 1, 16384, 2048, CppMicroGemmRVVBF16M1, (1, 32, 1)),
            ("decode_down", 1, 2048, 8192, CppMicroGemmRVVBF16M1, (1, 32, 1)),
            (
                "decode_large_projection",
                1,
                128256,
                2048,
                CppMicroGemmRVVBF16M1,
                (1, 32, 1),
            ),
        ]
        with (
            patch.object(
                cpu_vec_isa, "pick_vec_isa", return_value=cpu_vec_isa.VecRVV()
            ),
            patch(
                "torch._inductor.codegen.cpp_micro_gemm.pick_vec_isa",
                return_value=cpu_vec_isa.VecRVV(),
            ),
            V.set_graph_handler(FakeGraph()),
        ):
            for shape in representative_linear_shapes:
                name, m, n, k, expected_cls, expected_blocking = shape
                with self.subTest(name=name):
                    micro_gemm = create_micro_gemm(
                        "micro_gemm",
                        m,
                        n,
                        k,
                        input_dtype=torch.bfloat16,
                        input2_dtype=torch.bfloat16,
                        output_dtype=torch.float,
                        compute_dtype=torch.float,
                        alpha=1,
                        num_threads=8,
                        use_ref=False,
                    )
                    self.assertIsInstance(micro_gemm, expected_cls)
                    self.assertEqual(micro_gemm.register_blocking, expected_blocking)
                    source = micro_gemm.codegen_define(FakeKernel())
                    self.assertIn("#include <riscv_vector.h>", source)
                    self.assertIn("__riscv_vsetvl", source)
                    self.assertIn("__riscv_vle16", source)
                    self.assertIn("__riscv_vfmacc", source)
                    self.assertNotIn("extern_kernels.mm", source)
                    if expected_cls is CppMicroGemmRVVBF16M1:
                        self.assertIn("CppMicroGemmRVVBF16M1 requires M=1", source)
                        self.assertIn("__riscv_vfmacc_vv_f32m4_tu", source)


if __name__ == "__main__":
    run_tests()
