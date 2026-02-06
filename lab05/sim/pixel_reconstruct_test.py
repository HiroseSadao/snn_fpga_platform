import cocotb
import os, sys
from pathlib import Path
from cocotb.clock import Clock
from cocotb.runner import get_runner
from cocotb.triggers import RisingEdge, ClockCycles, ReadOnly

test_file = os.path.basename(__file__).replace(".py", "")

async def pclk_rise(dut):
    """Generate one camera_pclk rising edge aligned to clk (sample point for DUT)."""
    await RisingEdge(dut.clk)
    dut.camera_pclk.value = 1
    await RisingEdge(dut.clk)
    dut.camera_pclk.value = 0

async def drive_pixel_and_wait_valid(dut, pix):
    """(上位→下位) の 2 pclk を駆動し、pixel_valid 立上りで値を読む。"""
    hi = (pix >> 8) & 0xFF
    lo = pix & 0xFF

    # upper byte
    dut.camera_data.value = hi
    await pclk_rise(dut)

    # lower byte → このサンプルで pixel が完成し、同サイクルに pixel_valid が立つ
    dut.camera_data.value = lo
    await pclk_rise(dut)

    # pixel_valid 立上り“そのサイクル”で読み取る
    await RisingEdge(dut.pixel_valid)
    # await ReadOnly()  # 同サイクルの安定化

    got  = int(dut.pixel_data.value)
    hcnt = int(dut.pixel_h_count.value)
    vcnt = int(dut.pixel_v_count.value)
    return got, hcnt, vcnt

@cocotb.test()
async def pixel_reconstruct_basic_test(dut):
    """Correctness test matching the provided 'good' waveform:
       - 最初の pixel_valid 立上り時の hcount は 0
       - 以後、各 pixel_valid で hcount が 0,1,2,… と進む
       - 行末は hsync 1→0 を pclk で取り込ませて hcount=0 に戻り、vcount が +1
       - フレーム末は vsync=0 を pclk で取り込ませる
    """

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Reset & init
    dut.rst.value = 1
    dut.camera_pclk.value = 0
    dut.camera_h_sync.value = 0
    dut.camera_v_sync.value = 0
    dut.camera_data.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0

    dut._log.info("Starting pixel data simulation to match the reference waveform...")

    # ===== テスト条件 =====
    rows = 3
    pixel_values = [0xABCD, 0x1234, 0xFFFF, 0x00F0]  # 1行あたりのピクセル列
    cols = len(pixel_values)

    expected_v = 0

    for r in range(rows):
        expected_h = 0

        # 行開始：active 期間にする
        dut.camera_v_sync.value = 1
        dut.camera_h_sync.value = 1

        for i, pix in enumerate(pixel_values):
            got, hcnt, vcnt = await drive_pixel_and_wait_valid(dut, pix)

            # 参照波形のルールに合わせた検証
            assert got == pix, f"Pixel mismatch: exp {hex(pix)}, got {hex(got)}"
            assert hcnt == expected_h, f"H count mismatch at row {r}, pix {i}: exp {expected_h}, got {hcnt}"
            assert vcnt == expected_v, f"V count mismatch at row {r}, pix {i}: exp {expected_v}, got {vcnt}"

            dut._log.info(f"✓ pixel {hex(got)} @ (h={hcnt}, v={vcnt})")
            expected_h += 1

        # 行終端：hsync を 1→0 とし、その状態を pclk 立上りで取り込ませる
        dut.camera_h_sync.value = 0
        await pclk_rise(dut)             # DUT が「行終端」をサンプリング
        dut.camera_h_sync.value = 1      # 次の行へ戻す

        expected_v += 1                  # 次の行へ
        # 次の行の最初の pixel_valid では hcount==0 を期待（expected_h はループ先頭で0にセット）

    # フレーム終端：vsync を 0 にして pclk 立上りで取り込ませる
    dut.camera_v_sync.value = 0
    await pclk_rise(dut)

    await ClockCycles(dut.clk, 3)
    dut._log.info("✓ All pixels and (h,v) counts verified against the reference waveform!")

# -------- runner --------
def pixel_reconstruct_runner():
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(proj_path / "sim" / "model"))
    sources = [proj_path / "hdl" / "pixel_reconstruct.sv"]

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="pixel_reconstruct",
        parameters={},
        timescale=('1ns', '1ps'),
        waves=True,
        always=True
    )
    runner.test(hdl_toplevel="pixel_reconstruct", test_module=test_file, test_args=[])

if __name__ == "__main__":
    pixel_reconstruct_runner()
