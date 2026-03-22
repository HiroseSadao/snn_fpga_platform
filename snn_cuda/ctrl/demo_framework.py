from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import use_fpga_calc as fpga

from demo_compiler import DemoCompilePlan


@dataclass(frozen=True)
class DemoSessionInfo:
    backend_name: str
    caps: int
    port: str


class DemoBackend(Protocol):
    name: str

    def query_session_info(self, handle: object, plan: DemoCompilePlan) -> DemoSessionInfo:
        ...

    def run_plan(self, handle: object, plan: DemoCompilePlan, session: DemoSessionInfo) -> dict[str, object]:
        ...


class FpgaDemoBackend:
    name = "fpga_uart"

    def open_for_port(self, port: str) -> fpga.serial.Serial:
        if fpga.serial is None:
            raise RuntimeError(
                "pyserial is not installed. Install it to use FPGA communication paths."
            )
        ser = fpga.serial.Serial(
            port,
            fpga.BAUD,
            timeout=fpga.TIMEOUT_SEC,
            write_timeout=fpga.WRITE_TIMEOUT_SEC,
        )
        fpga.time.sleep(0.05)
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        return ser

    def query_session_info(self, handle: fpga.serial.Serial, plan: DemoCompilePlan) -> DemoSessionInfo:
        caps = fpga.fpga_train_query_caps(handle)
        return DemoSessionInfo(
            backend_name=self.name,
            caps=int(caps),
            port=plan.spec.execution.port,
        )

    def run_plan(
        self,
        handle: fpga.serial.Serial,
        plan: DemoCompilePlan,
        session: DemoSessionInfo,
    ) -> dict[str, object]:
        runtime_args = plan.to_runtime_args()
        if plan.batch_train_only:
            return self._run_train_only(handle, runtime_args, plan, session)
        return self._run_train_then_infer(handle, runtime_args, plan, session)

    def _base_result(self, plan: DemoCompilePlan, session: DemoSessionInfo) -> dict[str, object]:
        spec = plan.spec
        return {
            "status": "ok",
            "backend": session.backend_name,
            "mode": spec.execution.mode,
            "port": session.port,
            "caps_hex": f"0x{int(session.caps):08X}",
            "compiler_plan": plan.to_dict(),
            "network_name": spec.network.name,
            "dataset_name": spec.dataset.name,
            "start_lba": int(spec.dataset.start_lba),
            "seed": int(spec.execution.seed),
            "train_seed": int(spec.execution.seed if spec.execution.train_seed is None else spec.execution.train_seed),
            "infer_seed": int((spec.execution.seed + 1) if spec.execution.infer_seed is None else spec.execution.infer_seed),
            "timeout_sec": float(spec.execution.timeout_sec),
            "train_samples": int(spec.execution.train_samples),
            "infer_samples": int(spec.execution.infer_samples),
        }

    def _run_train_only(
        self,
        ser: fpga.serial.Serial,
        runtime_args: object,
        plan: DemoCompilePlan,
        session: DemoSessionInfo,
    ) -> dict[str, object]:
        fpga.fpga_batch_train_only(ser, runtime_args)
        elapsed_cycles = fpga.fpga_batch_read_summary_field_u64(ser, 7, 21)
        cycle_breakdown = fpga.fpga_batch_read_cycle_breakdown(ser)
        label_counts = fpga.fpga_read_train_label_counts_all(ser)

        result = self._base_result(plan, session)
        result.update(
            {
                "elapsed_cycles": int(elapsed_cycles),
                "elapsed_sec": float(fpga.cycles_to_seconds(elapsed_cycles)),
                "cycle_breakdown": {k: int(v) for k, v in cycle_breakdown.items()},
                "label_count_sum": int(fpga.np.sum(label_counts)),
                "nonzero_label_indices": [int(v) for v in fpga.np.nonzero(label_counts)[0]],
            }
        )
        return result

    def _run_train_then_infer(
        self,
        ser: fpga.serial.Serial,
        runtime_args: object,
        plan: DemoCompilePlan,
        session: DemoSessionInfo,
    ) -> dict[str, object]:
        fpga.fpga_batch_train_then_infer(ser, runtime_args)
        correct = fpga.fpga_batch_read_summary_field(ser, 6)
        elapsed_cycles = fpga.fpga_batch_read_summary_field_u64(ser, 7, 21)
        cycle_breakdown = fpga.fpga_batch_read_cycle_breakdown(ser)
        infer_samples = int(plan.spec.execution.infer_samples)
        accuracy = (float(correct) / float(infer_samples)) if infer_samples > 0 else 0.0

        result = self._base_result(plan, session)
        result.update(
            {
                "correct": int(correct),
                "accuracy": float(accuracy),
                "elapsed_cycles": int(elapsed_cycles),
                "elapsed_sec": float(fpga.cycles_to_seconds(elapsed_cycles)),
                "cycle_breakdown": {k: int(v) for k, v in cycle_breakdown.items()},
            }
        )
        return result


class DemoRunner:
    def __init__(self, backend: FpgaDemoBackend, plan: DemoCompilePlan):
        self.backend = backend
        self.plan = plan

    def run(self) -> tuple[DemoSessionInfo, dict[str, object]]:
        with self.backend.open_for_port(self.plan.spec.execution.port) as handle:
            session = self.backend.query_session_info(handle, self.plan)
            result = self.backend.run_plan(handle, self.plan, session)
        return session, result
