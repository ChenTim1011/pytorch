# Owner(s): ["module: inductor"]
import importlib
import platform
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from torch._C import FileCheck
from torch._inductor import config, cpu_vec_isa
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_cpp_code
from torch.testing._internal.common_device_type import instantiate_device_type_tests
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


class TestRvvTemplateRouting(TestCase):
    hw_classification = HardwareClassification.CPU

    def _uses_cpp_template(self, isa, m=4, n=1024, k=1024):
        from types import SimpleNamespace

        from torch._inductor import ir
        from torch._inductor.codegen import cpp_micro_gemm
        from torch._inductor.sizevars import SizeVarAllocator
        from torch._inductor.utils import use_cpp_gemm_template
        from torch._inductor.virtualized import V

        device = torch.device("cpu")
        dtype = torch.bfloat16
        a = ir.TensorBox.create(
            ir.InputBuffer(name="x", layout=ir.FixedLayout(device, dtype, [m, k]))
        )
        b = ir.TensorBox.create(
            ir.ConstantBuffer(name="w", layout=ir.FixedLayout(device, dtype, [k, n]))
        )
        layout = ir.FixedLayout(device, dtype, [m, n])
        graph = SimpleNamespace(sizevars=SizeVarAllocator(), constants={"w": None})
        with (
            V.set_graph_handler(graph),
            patch.object(cpp_micro_gemm, "pick_vec_isa", return_value=isa),
        ):
            return use_cpp_gemm_template(layout, a, b)

    @config.patch(
        {
            "max_autotune": False,
            "max_autotune_gemm": False,
            "max_autotune_gemm_backends": "CPP",
            "cpp.weight_prepack": True,
        }
    )
    def test_non_rvv_default_keeps_cpp_template_opt_in(self):
        self.assertFalse(self._uses_cpp_template(cpu_vec_isa.VecAVX2()))

    @config.patch(
        {
            "max_autotune": False,
            "max_autotune_gemm": True,
            "max_autotune_gemm_backends": "ATEN",
            "cpp.weight_prepack": True,
        }
    )
    def test_rvv_autotune_honors_backend_filter(self):
        self.assertFalse(self._uses_cpp_template(cpu_vec_isa.VecRVV()))

    @config.patch(
        {
            "max_autotune": False,
            "max_autotune_gemm": False,
            "cpp.weight_prepack": True,
            "cpp.vec_isa_ok": True,
            "cpp.simdlen": 256,
            "compile_threads": 1,
        }
    )
    def test_rvv_default_codegen_without_target_benchmark(self):
        self._check_default_codegen(has_template=True)

    def test_rvv_default_falls_back_without_template(self):
        self._check_default_codegen(has_template=False)

    @config.patch(
        {
            "max_autotune": False,
            "max_autotune_gemm": False,
            "cpp.weight_prepack": True,
            "cpp.vec_isa_ok": True,
            "cpp.simdlen": 256,
            "compile_threads": 1,
        }
    )
    def _check_default_codegen(self, has_template):
        from torch._inductor import codecache, cpp_builder
        from torch._inductor.codegen.cpp_gemm_template import CppGemmTemplate

        class MM(nn.Module):
            def forward(self, a, b):
                return torch.mm(a, b)

        a = torch.randn(1, 1024, dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, dtype=torch.bfloat16)
        exported = torch.export.export(MM(), (a, b))
        self.addCleanup(cpu_vec_isa.valid_vec_isa_list.cache_clear)
        for backends in ("ATEN", "ATEN,TRITON,CPP"):
            with self.subTest(backends=backends):
                cpu_vec_isa.valid_vec_isa_list.cache_clear()
                with (
                    config.patch({"max_autotune_gemm_backends": backends}),
                    nullcontext()
                    if has_template
                    else patch.object(
                        CppGemmTemplate, "add_choices", return_value=None
                    ),
                    patch.object(
                        cpu_vec_isa.platform, "machine", return_value="riscv64"
                    ),
                    patch.object(
                        torch.backends.cpu, "get_cpu_capability", return_value="RVV"
                    ),
                    patch.object(
                        codecache.AotCodeCompiler, "compile", return_value="source_only"
                    ) as backend,
                    patch.object(
                        cpp_builder.CppBuilder,
                        "build",
                        side_effect=AssertionError(
                            "default mode must not benchmark target kernels"
                        ),
                    ) as build,
                ):
                    self.assertEqual(
                        torch._inductor.aot_compile(exported.graph_module, (a, b)),
                        "source_only",
                    )
                backend.assert_called_once()
                build.assert_not_called()
                kernel = backend.call_args.args[2]
                if has_template:
                    self.assertIn("#include <riscv_vector.h>", kernel)
                    self.assertIn("__riscv_vfmacc", kernel)
                else:
                    self.assertNotIn("__riscv_vfmacc", kernel)
                    self.assertIn("mm_out", backend.call_args.args[1])

    def test_rvv_bf16_template_profitability_boundaries(self):
        from torch._inductor.utils import _is_profitable_rvv_bf16_template_shape

        self.assertTrue(
            _is_profitable_rvv_bf16_template_shape(1, 1024, 1024, 1_048_576)
        )
        self.assertTrue(
            _is_profitable_rvv_bf16_template_shape(128, 1024, 1024, 134_217_728)
        )
        self.assertFalse(
            _is_profitable_rvv_bf16_template_shape(129, 1024, 1024, 135_266_304)
        )
        self.assertFalse(
            _is_profitable_rvv_bf16_template_shape(1, 1023, 1024, 1_047_552)
        )
        self.assertFalse(
            _is_profitable_rvv_bf16_template_shape(1, 1024, 1023, 1_047_552)
        )
        self.assertFalse(_is_profitable_rvv_bf16_template_shape(1, 1024, 1024, 999_999))


class TestRvvTemplateExecution(TestCase):
    hw_classification = HardwareClassification.CPU

    @unittest.skipIf(platform.machine() != "riscv64", "RVV-only regression")
    @config.patch(
        {
            "max_autotune_gemm": False,
            "max_autotune_gemm_backends": "ATEN",
            "freezing": True,
            "cpp.weight_prepack": True,
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
        }
    )
    def test_rvv_bf16_micro_gemm_without_cpp_autotune_backend(self, device):
        from torch._inductor.cpu_vec_isa import pick_vec_isa

        if not isinstance(pick_vec_isa(), cpu_vec_isa.VecRVV):
            self.skipTest("VecRVV is not selected")

        class LinearMM(nn.Module):
            def __init__(self, n: int, k: int) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(n, k, dtype=torch.bfloat16, device=device)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.mm(x, self.weight.t())

        model = LinearMM(3072, 2048).eval()
        x = torch.randn(1, 2048, dtype=torch.bfloat16, device=device)
        with torch.inference_mode():
            expected = model(x)
            actual, code = run_and_get_cpp_code(
                torch.compile(model, backend="inductor", fullgraph=True), x
            )

        self.assertEqual(actual, expected, rtol=2e-2, atol=2e-2)
        FileCheck().check("#include <riscv_vector.h>").check("kernel_micro_gemm").check(
            "__riscv_vsetvl"
        ).check("__riscv_vle16").check("__riscv_vfmacc").check_not(
            "extern_kernels.mm"
        ).run(code)

        torch._dynamo.reset()
        small_model = LinearMM(64, 64).eval()
        small_x = torch.randn(1, 64, dtype=torch.bfloat16, device=device)
        with torch.inference_mode():
            _, small_code = run_and_get_cpp_code(
                torch.compile(small_model, backend="inductor", fullgraph=True),
                small_x,
            )
        FileCheck().check_not("kernel_micro_gemm").check_not(
            "#include <riscv_vector.h>"
        ).run(small_code)

    @unittest.skipIf(platform.machine() != "riscv64", "RVV-only regression")
    @config.patch(
        {
            "max_autotune_gemm": False,
            "max_autotune_gemm_backends": "ATEN",
            "cpp.weight_prepack": True,
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
        }
    )
    def test_rvv_bf16_linear_runtime_weight_reuse(self, device):
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch._inductor.cpu_vec_isa import pick_vec_isa

        if not isinstance(pick_vec_isa(), cpu_vec_isa.VecRVV):
            self.skipTest("VecRVV is not selected")

        def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return F.linear(x, weight, None)

        counter = CompileCounterWithBackend("inductor")
        opt_linear = torch.compile(
            linear, backend=counter, dynamic=False, fullgraph=True
        )
        x = torch.randn(16, 1024, dtype=torch.bfloat16, device=device)
        weights = [
            nn.Parameter(torch.randn(1024, 1024, dtype=torch.bfloat16, device=device))
            for _ in range(3)
        ]

        with torch.inference_mode():
            for weight in weights:
                actual = opt_linear(x, weight)
                expected = linear(x, weight)
                self.assertEqual(actual, expected, rtol=2e-2, atol=1.0)

        self.assertEqual(counter.frame_count, 1)

        torch._dynamo.reset()
        with torch.inference_mode():
            _, code = run_and_get_cpp_code(
                torch.compile(linear, backend="inductor", fullgraph=True),
                x,
                weights[0],
            )
        FileCheck().check("#include <riscv_vector.h>").check("kernel_micro_gemm").check(
            "__riscv_vsetvl"
        ).check("load_bf16_as_f32").check("__riscv_vfmacc").check_not(
            "extern_kernels.mm"
        ).run(code)

    @unittest.skipIf(platform.machine() != "riscv64", "RVV-only regression")
    @config.patch(
        {
            "max_autotune_gemm": False,
            "max_autotune_gemm_backends": "ATEN",
            "cpp.weight_prepack": True,
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
        }
    )
    @torch._dynamo.config.patch("inline_single_use_invoke_subgraph", False)
    def test_rvv_bf16_linear_nested_compile_region(self, device):
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch._inductor.cpu_vec_isa import pick_vec_isa

        if not isinstance(pick_vec_isa(), cpu_vec_isa.VecRVV):
            self.skipTest("VecRVV is not selected")

        @torch.compiler.nested_compile_region
        def linear_region(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return F.linear(x, weight, None)

        def model(
            x: torch.Tensor,
            w0: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
        ) -> torch.Tensor:
            return linear_region(x, w0) + linear_region(x, w1) + linear_region(x, w2)

        x = torch.randn(16, 1024, dtype=torch.bfloat16, device=device)
        weights = [
            nn.Parameter(torch.randn(1024, 1024, dtype=torch.bfloat16, device=device))
            for _ in range(3)
        ]

        counter = CompileCounterWithBackend("inductor")
        opt_model = torch.compile(model, backend=counter, dynamic=False, fullgraph=True)
        with torch.inference_mode():
            actual = opt_model(x, *weights)
            expected = model(x, *weights)
            self.assertEqual(actual, expected, rtol=2e-2, atol=3.0)

        self.assertEqual(counter.frame_count, 1)

        torch._dynamo.reset()
        with torch.inference_mode():
            _, code = run_and_get_cpp_code(
                torch.compile(model, backend="inductor", fullgraph=True),
                x,
                *weights,
            )
        FileCheck().check("#include <riscv_vector.h>").check("kernel_micro_gemm").check(
            "__riscv_vsetvl"
        ).check("load_bf16_as_f32").check("__riscv_vfmacc").check_not(
            "extern_kernels.mm"
        ).run(code)

    @unittest.skipIf(platform.machine() != "riscv64", "RVV-only regression")
    @config.patch(
        {
            "max_autotune_gemm": False,
            "max_autotune_gemm_backends": "ATEN",
            "cpp.weight_prepack": True,
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
        }
    )
    @torch._dynamo.config.patch("inline_single_use_invoke_subgraph", False)
    def test_rvv_bf16_representative_linear_buckets(self, device):
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch._inductor.cpu_vec_isa import pick_vec_isa

        if not isinstance(pick_vec_isa(), cpu_vec_isa.VecRVV):
            self.skipTest("VecRVV is not selected")

        @torch.compiler.nested_compile_region
        def linear_region(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return F.linear(x, weight, None)

        def repeated_linear_model(
            x: torch.Tensor,
            w0: torch.Tensor,
            w1: torch.Tensor,
        ) -> torch.Tensor:
            return linear_region(x, w0) + linear_region(x, w1)

        representative_buckets = [
            ("decode_m1_qkv", 1, 3072, 2048, 2e-2, 4.0),
            ("prefill_m64_qkv", 64, 3072, 2048, 2e-2, 4.0),
            ("prefill_m128_qkv", 128, 3072, 2048, 2e-2, 4.0),
        ]

        for name, m, n, k, rtol, atol in representative_buckets:
            with self.subTest(name=name):
                torch._dynamo.reset()
                x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
                weights = [
                    nn.Parameter(torch.randn(n, k, dtype=torch.bfloat16, device=device))
                    for _ in range(2)
                ]

                counter = CompileCounterWithBackend("inductor")
                opt_model = torch.compile(
                    repeated_linear_model,
                    backend=counter,
                    dynamic=False,
                    fullgraph=True,
                )
                with torch.inference_mode():
                    actual = opt_model(x, *weights)
                    expected = repeated_linear_model(x, *weights)
                    self.assertEqual(actual, expected, rtol=rtol, atol=atol)

                self.assertEqual(counter.frame_count, 1)

                torch._dynamo.reset()
                with torch.inference_mode():
                    _, code = run_and_get_cpp_code(
                        torch.compile(
                            repeated_linear_model,
                            backend="inductor",
                            fullgraph=True,
                        ),
                        x,
                        *weights,
                    )
                self.assertIn("#include <riscv_vector.h>", code)
                self.assertIn("kernel_micro_gemm", code)
                self.assertIn("__riscv_vsetvl", code)
                self.assertIn("__riscv_vle16", code)
                self.assertIn("__riscv_vfmacc", code)
                self.assertNotIn("extern_kernels.mm", code)


instantiate_device_type_tests(TestRvvTemplateExecution, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
