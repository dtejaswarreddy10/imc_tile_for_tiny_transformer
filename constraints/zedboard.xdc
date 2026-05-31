# ---------------------------------------------------------------------------
# Zedboard constraints (Zynq-7020, xc7z020clg484-1)
# Pinout reference: Digilent Zedboard schematic rev. D.
# ---------------------------------------------------------------------------

# 100 MHz system clock (GCLK)
set_property PACKAGE_PIN Y9       [get_ports clk_in]
set_property IOSTANDARD  LVCMOS33 [get_ports clk_in]
create_clock -name sys_clk -period 10.000 [get_ports clk_in]

# LD0 -- visual "done" indicator
set_property PACKAGE_PIN T22      [get_ports led]
set_property IOSTANDARD  LVCMOS33 [get_ports led]
