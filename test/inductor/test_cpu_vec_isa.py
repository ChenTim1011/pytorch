# Owner(s): ["module: inductor"]
import ctypes
import os
import platform
import shlex
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import torch
from torch._inductor import codecache, config, cpp_builder, cpu_vec_isa
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


class TestVecISA(TestCase):
    hw_classification = HardwareClassification.CPU

    def setUp(self):
        super().setUp()
        cpu_vec_isa.valid_vec_isa_list.cache_clear()
        self.addCleanup(cpu_vec_isa.valid_vec_isa_list.cache_clear)

    @unittest.skipUnless(sys.platform == "linux", "RVV build flags require Linux")
    @unittest.skipIf(config.is_fbcode(), "external compiler build options")
    @config.patch({"cpp.march": None})
    def test_rvv_build_command_preserves_vector_arch(self):
        compiler = cpp_builder.get_cpp_compiler()
        with (
            mock.patch.object(cpu_vec_isa.platform, "machine", return_value="riscv64"),
            tempfile.TemporaryDirectory() as tmp,
        ):
            options = cpp_builder.CppTorchOptions(
                vec_isa=cpu_vec_isa.VecRVV(), compiler=compiler
            )
            builder = cpp_builder.CppBuilder(
                name="rvv_flags",
                sources="rvv_flags.cpp",
                BuildOption=options,
                output_dir=tmp,
            )
            args = shlex.split(builder.get_command_line())
        march_flags = [arg for arg in args if arg.startswith("-march=")]
        self.assertTrue(march_flags)
        self.assertEqual(march_flags[-1], "-march=rv64gcv_zvl128b")
        self.assertIn("CPU_CAPABILITY_RVV", options.get_definitions())
        self.assertIn("CPU_CAPABILITY=RVV", options.get_definitions())
        self.assertIn("HAVE_RVV_CPU_DEFINITION", options.get_definitions())

    @unittest.skipUnless(sys.platform == "linux", "RVV targets Linux")
    def test_rvv_gcc_skips_precompiled_header(self):
        with (
            mock.patch.object(cpp_builder, "is_gcc", return_value=True),
            mock.patch.object(
                codecache,
                "CppBuilder",
                side_effect=AssertionError("RVV GCC must use textual headers"),
            ),
        ):
            self.assertIsNone(
                codecache._precompile_header(
                    "torch/csrc/inductor/cpp_prefix.h",
                    "rvv_gcc_pch",
                    vec_isa=cpu_vec_isa.VecRVV(),
                    device_type="cpu",
                )
            )

    @unittest.skipUnless(sys.platform == "linux", "GCC PCH requires Linux")
    @unittest.skipIf(config.is_fbcode(), "external compiler build options")
    def test_scalar_gcc_keeps_precompiled_header(self):
        if not cpp_builder.is_gcc():
            self.skipTest("requires GCC")
        with tempfile.TemporaryDirectory() as tmp:
            header = os.path.join(tmp, "probe.h")
            with open(header, "w") as f:
                f.write("inline int pch_value() { return 7; }\n")
            pch = codecache._precompile_header(
                header,
                "scalar_gcc_pch",
                vec_isa=cpu_vec_isa.invalid_vec_isa,
                device_type="cpu",
            )
            if pch is None:
                raise AssertionError("scalar GCC must retain PCH support")
            self.assertTrue(os.path.isfile(pch + ".gch"))
            source = os.path.join(tmp, "consumer.cpp")
            with open(source, "w") as f:
                f.write('extern "C" int pch_probe() { return pch_value(); }\n')
            options = cpp_builder.CppTorchDeviceOptions(
                vec_isa=cpu_vec_isa.invalid_vec_isa,
                device_type="cpu",
                extra_flags=("-Werror=invalid-pch",),
            )
            options.precompiled_header = pch
            builder = cpp_builder.CppBuilder(
                name="pch_consumer", sources=source, BuildOption=options, output_dir=tmp
            )
            builder.build()
            lib = ctypes.CDLL(builder.get_target_file_path())
            self.assertEqual(lib.pch_probe(), 7)

    @parametrize("capability", ["RVV", "DEFAULT"])
    @parametrize("probe_ok", [False, True])
    def test_rvv_capability_enables_vector_isa(self, capability, probe_ok):
        with (
            mock.patch.object(cpu_vec_isa.sys, "platform", "linux"),
            mock.patch.object(cpu_vec_isa.platform, "machine", return_value="riscv64"),
            mock.patch.object(
                torch.backends.cpu,
                "get_cpu_capability",
                return_value=capability,
            ),
            mock.patch.object(
                cpu_vec_isa.VecISA, "__bool__", return_value=probe_ok
            ) as probe,
        ):
            self.assertEqual(
                [str(isa) for isa in cpu_vec_isa.valid_vec_isa_list()],
                ["rvv"] if capability == "RVV" and probe_ok else [],
            )
            if capability != "RVV":
                probe.assert_not_called()

    @parametrize("system", ["darwin", "win32"])
    def test_rvv_requires_linux(self, system):
        with (
            mock.patch.object(cpu_vec_isa.sys, "platform", system),
            mock.patch.object(cpu_vec_isa.platform, "machine", return_value="riscv64"),
            mock.patch.object(
                cpu_vec_isa.platform, "processor", return_value="riscv64"
            ),
            mock.patch.object(
                torch.backends.cpu,
                "get_cpu_capability",
                return_value="RVV",
            ),
            mock.patch.object(cpu_vec_isa.VecISA, "__bool__", return_value=True),
        ):
            self.assertEqual(cpu_vec_isa.valid_vec_isa_list(), [])

    @parametrize(
        "capability,expected", [("rvv", "rvv"), ("default", "INVALID_VEC_ISA")]
    )
    @config.patch({"cpp.simdlen": None})
    def test_rvv_capability_override(self, capability, expected):
        with (
            mock.patch.object(cpu_vec_isa.sys, "platform", "linux"),
            mock.patch.object(cpu_vec_isa.platform, "machine", return_value="riscv64"),
            mock.patch.object(
                torch.backends.cpu, "get_cpu_capability", return_value="RVV"
            ),
            mock.patch.object(cpu_vec_isa.VecISA, "__bool__", return_value=True),
            mock.patch.dict(os.environ, {"ATEN_CPU_CAPABILITY": capability}),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            self.assertEqual(str(cpu_vec_isa.pick_vec_isa()), expected)
            self.assertEqual([str(w.message) for w in caught], [])

    @parametrize("simdlen", [0, 1, 128, 256, 512])
    def test_rvv_simdlen(self, simdlen):
        with (
            mock.patch.object(cpu_vec_isa.sys, "platform", "linux"),
            mock.patch.object(cpu_vec_isa.platform, "machine", return_value="riscv64"),
            mock.patch.object(
                torch.backends.cpu, "get_cpu_capability", return_value="RVV"
            ),
            mock.patch.object(cpu_vec_isa.VecISA, "__bool__", return_value=True),
            config.patch({"cpp.simdlen": simdlen}),
        ):
            self.assertEqual(
                str(cpu_vec_isa.pick_vec_isa()),
                "rvv" if simdlen == 256 else "INVALID_VEC_ISA",
            )

    @unittest.skipIf(config.is_fbcode(), "fbcode skips the ISA build probe")
    @config.patch({"cpp.vec_isa_ok": None})
    def test_subclass_build_probe(self):
        class ArithmeticOnlyISA(cpu_vec_isa.VecISA):
            _avx_code = "ISA-specific arithmetic probe"

            def __str__(self):
                return "arithmetic_only"

        isa = ArithmeticOnlyISA()
        # Model a toolchain that supports this ISA's arithmetic probe but
        # cannot build the base probe's optional math-library operations.
        with mock.patch.object(
            isa,
            "check_build",
            side_effect=lambda code: code == ArithmeticOnlyISA._avx_code,
        ):
            self.assertTrue(bool(isa))

    @unittest.skipIf(config.is_fbcode(), "fbcode skips the ISA build probe")
    @config.patch({"cpp.vec_isa_ok": None})
    def test_inherited_build_probe(self):
        class DefaultProbeISA(cpu_vec_isa.VecISA):
            def __str__(self):
                return "default_probe"

        isa = DefaultProbeISA()
        with mock.patch.object(
            isa,
            "check_build",
            side_effect=lambda code: code == cpu_vec_isa.VecISA._avx_code,
        ):
            self.assertTrue(bool(isa))

    @unittest.skipIf(config.is_fbcode(), "fbcode skips the ISA build probe")
    @config.patch({"cpp.vec_isa_ok": None})
    def test_failed_build_disables_isa(self):
        class UnsupportedISA(cpu_vec_isa.VecISA):
            def __str__(self):
                return "unsupported"

        isa = UnsupportedISA()
        with mock.patch.object(isa, "check_build", return_value=False):
            self.assertFalse(bool(isa))

    @parametrize("override", [False, True])
    def test_override_skips_build_probe(self, override):
        class OverrideISA(cpu_vec_isa.VecISA):
            def __str__(self):
                return "override"

        isa = OverrideISA()
        with (
            config.patch({"cpp.vec_isa_ok": override}),
            mock.patch.object(
                isa,
                "check_build",
                side_effect=AssertionError("an explicit override must skip the probe"),
            ),
        ):
            self.assertEqual(bool(isa), override)


instantiate_parametrized_tests(TestVecISA)


class TestRvvSourceCodegen(InductorTestCase):
    hw_classification = HardwareClassification.CPU

    @unittest.skipUnless(sys.platform == "linux", "RVV codegen targets Linux")
    @unittest.skipIf(config.is_fbcode(), "external compiler build options")
    @parametrize(
        "dtype,cpp_type",
        [
            subtest((torch.float32, "float"), name="fp32"),
            subtest((torch.bfloat16, "at::BFloat16"), name="bf16"),
            subtest((torch.float16, "at::Half"), name="fp16"),
        ],
    )
    @parametrize(
        "capability,probe_ok,use_vector",
        [
            subtest(("RVV", True, True), name="rvv"),
            subtest(("RVV", False, False), name="build_rejected"),
            subtest(("DEFAULT", True, False), name="aten_unsupported"),
        ],
    )
    @config.patch({"cpp.vec_isa_ok": True, "cpp.simdlen": 256, "compile_threads": 1})
    def test_pointwise_codegen_without_target_execution(
        self, dtype, cpp_type, capability, probe_ok, use_vector
    ):
        class Pointwise(torch.nn.Module):
            def forward(self, x):
                return x * x + 1

        x = torch.randn(4099, dtype=dtype)
        exported = torch.export.export(Pointwise(), (x,))
        cpu_vec_isa.valid_vec_isa_list.cache_clear()
        self.addCleanup(cpu_vec_isa.valid_vec_isa_list.cache_clear)
        with (
            config.patch({"cpp.vec_isa_ok": probe_ok}),
            mock.patch.object(cpu_vec_isa.platform, "machine", return_value="riscv64"),
            mock.patch.object(
                torch.backends.cpu, "get_cpu_capability", return_value=capability
            ),
            mock.patch.object(
                codecache.AotCodeCompiler, "compile", return_value="source_only_test"
            ) as backend,
            mock.patch.object(
                cpp_builder.CppBuilder,
                "build",
                side_effect=AssertionError(
                    "codegen must stop before binary compilation"
                ),
            ) as build,
        ):
            self.assertEqual(
                str(cpu_vec_isa.pick_vec_isa()),
                "rvv" if use_vector else "INVALID_VEC_ISA",
            )
            result = torch._inductor.aot_compile(exported.graph_module, (x,))
        backend.assert_called_once()
        build.assert_not_called()
        self.assertEqual(result, "source_only_test")
        wrapper, kernel = backend.call_args.args[1:3]
        self.assertTrue(wrapper)
        if use_vector:
            self.assertIn("at::vec::Vectorized", kernel)
        else:
            self.assertNotIn("at::vec::Vectorized", kernel)
        self.assertIn(cpp_type, kernel)


instantiate_parametrized_tests(TestRvvSourceCodegen)


class TestRvvCodegen(InductorTestCase):
    hw_classification = HardwareClassification.CPU

    @onlyCPU
    @unittest.skipUnless(
        platform.machine() == "riscv64"
        and torch.backends.cpu.get_cpu_capability() == "RVV",
        "requires an RVV-enabled RISC-V PyTorch build",
    )
    @parametrize("size", [8, 9, 4096, 4099])
    @parametrize("precompile_headers", [False, True])
    @config.patch({"cpp.simdlen": 256, "compile_threads": 1})
    def test_pointwise(self, device, size, precompile_headers):
        def pointwise(x):
            return x * x + 1

        self.assertEqual(str(cpu_vec_isa.pick_vec_isa()), "rvv")
        x = torch.randn(size, dtype=torch.float32, device=device)
        with config.patch({"cpp_cache_precompile_headers": precompile_headers}):
            compiled = torch.compile(
                pointwise, backend="inductor", fullgraph=True, dynamic=False
            )
            actual, code = run_and_get_code(compiled, x)
        self.assertEqual(actual, pointwise(x))
        self.assertIn("at::vec::Vectorized<float>", "\n".join(code))


instantiate_device_type_tests(TestRvvCodegen, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
