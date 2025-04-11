# proc is a function. this is used later to connect all the vdd! and vss! together.
proc connectGlobalNets {} {
	globalNetConnect vdd! -type pgpin -pin vdd! -all
	globalNetConnect vss! -type pgpin -pin vss! -all
	globalNetConnect vdd! -type tiehi -all
	globalNetConnect vss! -type tielo -all
	applyGlobalNets
}

# set the top level module name (used elsewhere in the scripts)
set design_toplevel control

# set the verilog file to pnr
set init_verilog ../synth/outputs/$design_toplevel.v

# set the lef file of your standard cells
# when you add your regfile lef, it is here
# if you want to supply more than one lef use the following syntax:
# set init_lef_file "1.lef 2.lef"
set init_lef_file stdcells.lef

# actually set the top level cell name
set init_top_cell $design_toplevel

# set power and ground net names
set init_pwr_net vdd!
set init_gnd_net vss!

# set multi-mode multi-corner file
# this file contains the operating conditions used to evaluate timing
# for your design. In our case, we just use the single lib file as our corner.
# In ECE 498HK, this will contain slow, typical and fast corners
# for the wires and the standard cells
set init_mmmc_file mmmc.tcl

# actually init the design
init_design

# connect all the global nets in the design together (vdd!, vss!)
# the function is defined above.
connectGlobalNets

# TODO floorplan your design. Put the size of your chip that you want here.
floorPlan -site CoreSite -s 100 10 10 10 10 10

# create the horizontal vdd! and vss! wires used by the standard cells.
sroute -allowJogging 0 -allowLayerChange 0 -crossoverViaLayerRange { metal7 metal1 } -layerChangeRange { metal7 metal1 } -nets { vss! vdd! }

# create a power ring around your processor, connecting all the vss! and vdd! together physically.
addRing \
	-follow core \
	-offset {top 2 bottom 2 left 2 right 2} \
	-spacing {top 2 bottom 2 left 2 right 2} \
	-width {top 2 bottom 2 left 2 right 2} \
	-layer {top metal7 bottom metal7 left metal8 right metal8} \
	-nets { vss! vdd! }

# TODO add power grid
addStripe \
	-direction horizontal \
	-layer metal7 \
	-nets { vss! vdd! } \
	-spacing 1.6 \
	-width 0.4 \
	-set_to_set_distance 4
addStripe  \
	-direction vertical \
	-layer metal8 \
	-nets { vss! vdd! } \
	-spacing 1.6 \
	-width 0.4 \
	-set_to_set_distance 4

# TODO restrict routing to only metal 6
setDesignMode -topRoutingLayer metal6

# TODO for the regfile part, place the regfile marco
# placeInstance ...

# TODO specify where are the pins
# mem_mux
editPin \
	-assign {0.4 0} \
	-layer metal3 \
	-pin {mem_mux_sel_inv[0]} \
    -side INSIDE
editPin \
	-assign {1.035 0} \
	-layer metal3 \
	-pin {mem_mux_sel[0]} \
    -side INSIDE
editPin \
	-assign {2.75 0} \
	-layer metal3 \
	-pin {mem_mux_sel_inv[1]} \
    -side INSIDE
editPin \
	-assign {3.32 0} \
	-layer metal3 \
	-pin {mem_mux_sel[1]} \
    -side INSIDE
editPin \
	-assign {2.61 0} \
	-layer metal3 \
	-pin {mem_mux_sel_inv[2]} \
    -side INSIDE
editPin \
	-assign {1.975 0} \
	-layer metal3 \
	-pin {mem_mux_sel[2]} \
    -side INSIDE

# rd_mux
editPin \
	-assign {4.3375 0} \
	-layer metal3 \
	-pin {rd_mux_sel_inv[0]} \
    -side INSIDE
editPin \
	-assign {4.6625 0} \
	-layer metal3 \
	-pin {rd_mux_sel[0]} \
    -side INSIDE
editPin \
	-assign {5.0825 0} \
	-layer metal3 \
	-pin {cmp_out} \
    -side INSIDE
editPin \
	-assign {5.4275 0} \
	-layer metal3 \
	-pin {rd_mux_sel[1]} \
    -side INSIDE
editPin \
	-assign {5.5775 0} \
	-layer metal3 \
	-pin {rd_mux_sel_inv[1]} \
    -side INSIDE
editPin \
	-assign {7.2975 0} \
	-layer metal3 \
	-pin {rd_mux_sel[2]} \
    -side INSIDE
editPin \
	-assign {7.4975 0} \
	-layer metal3 \
	-pin {rd_mux_sel_inv[2]} \
    -side INSIDE

# modules after regfile
editPin \
	-assign {62.8675 0} \
	-layer metal3 \
	-pin {clk} \
    -side INSIDE
editPin \
	-assign {67.4225 0} \
	-layer metal3 \
	-pin {alu_inv_rs2} \
    -side INSIDE
editPin \
	-assign {68.455 0} \
	-layer metal3 \
	-pin {alu_mux_2_sel_inv} \
    -side INSIDE
editPin \
	-assign {68.605 0} \
	-layer metal3 \
	-pin {alu_mux_2_sel} \
    -side INSIDE
editPin \
	-assign {68.77 0} \
	-layer metal3 \
	-pin {cmp_mux_sel} \
    -side INSIDE
editPin \
	-assign {68.92 0} \
	-layer metal3 \
	-pin {cmp_mux_sel_inv} \
    -side INSIDE

# compare
editPin \
	-assign {72.03 0} \
	-layer metal3 \
	-pin {cmp_eq} \
    -side INSIDE
editPin \
	-assign {73.5 0} \
	-layer metal3 \
	-pin {cmp_lt} \
    -side INSIDE
editPin \
	-assign {73.74 0} \
	-layer metal3 \
	-pin {alu_mux_1_sel} \
    -side INSIDE
editPin \
	-assign {73.89 0} \
	-layer metal3 \
	-pin {alu_mux_1_sel_inv} \
    -side INSIDE

# alu
editPin \
	-assign {79.165 0} \
	-layer metal3 \
	-pin {alu_cin} \
    -side INSIDE
editPin \
	-assign {81.6425 0} \
	-layer metal3 \
	-pin {alu_op_inv[0]} \
    -side INSIDE
editPin \
	-assign {81.9575 0} \
	-layer metal3 \
	-pin {alu_op[0]} \
    -side INSIDE
editPin \
	-assign {82.7325 0} \
	-layer metal3 \
	-pin {alu_op[1]} \
    -side INSIDE
editPin \
	-assign {82.8825 0} \
	-layer metal3 \
	-pin {alu_op_inv[1]} \
    -side INSIDE

# pc
editPin \
	-assign {83.9575 0} \
	-layer metal3 \
	-pin {pc_mux_sel_inv} \
    -side INSIDE
editPin \
	-assign {84.1075 0} \
	-layer metal3 \
	-pin {pc_mux_sel} \
    -side INSIDE
editPin \
	-assign {84.2575 0} \
	-layer metal3 \
	-pin {rst} \
    -side INSIDE
editPin \
	-assign {84.5625 0} \
	-layer metal3 \
	-pin {rst_inv} \
    -side INSIDE

# shift
editPin \
	-assign {101.3550 0} \
	-layer metal3 \
	-pin {shift_dir} \
    -side INSIDE
editPin \
	-assign {101.565 0} \
	-layer metal3 \
	-pin {shift_dir_inv} \
    -side INSIDE
editPin \
	-assign {102.615 0} \
	-layer metal3 \
	-pin {shift_msb} \
    -side INSIDE

# regfile
# reg 0
editPin \
	-assign {8.405 0} \
	-layer metal3 \
	-pin {rs1_sel_inv[0]} \
    -side INSIDE
editPin \
	-assign {8.58 0} \
	-layer metal3 \
	-pin {rs1_sel[0]} \
    -side INSIDE
editPin \
	-assign {8.72 0} \
	-layer metal3 \
	-pin {rs2_sel_inv[0]} \
    -side INSIDE
editPin \
	-assign {8.86 0} \
	-layer metal3 \
	-pin {rs2_sel[0]} \
    -side INSIDE

# reg 1-31
editPin \
    -assign {9.212500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[1]} \
    -side INSIDE
editPin \
    -assign {9.465000 0} \
    -layer metal3 \
    -pin {rd_sel[1]} \
    -side INSIDE
editPin \
    -assign {9.977500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[1]} \
    -side INSIDE
editPin \
    -assign {10.117500 0} \
    -layer metal3 \
    -pin {rs1_sel[1]} \
    -side INSIDE
editPin \
    -assign {10.397500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[1]} \
    -side INSIDE
editPin \
    -assign {10.537500 0} \
    -layer metal3 \
    -pin {rs2_sel[1]} \
    -side INSIDE


editPin \
    -assign {10.940000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[2]} \
    -side INSIDE
editPin \
    -assign {11.192500 0} \
    -layer metal3 \
    -pin {rd_sel[2]} \
    -side INSIDE
editPin \
    -assign {11.705000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[2]} \
    -side INSIDE
editPin \
    -assign {11.845000 0} \
    -layer metal3 \
    -pin {rs1_sel[2]} \
    -side INSIDE
editPin \
    -assign {12.125000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[2]} \
    -side INSIDE
editPin \
    -assign {12.265000 0} \
    -layer metal3 \
    -pin {rs2_sel[2]} \
    -side INSIDE


editPin \
    -assign {12.667500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[3]} \
    -side INSIDE
editPin \
    -assign {12.920000 0} \
    -layer metal3 \
    -pin {rd_sel[3]} \
    -side INSIDE
editPin \
    -assign {13.432500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[3]} \
    -side INSIDE
editPin \
    -assign {13.572500 0} \
    -layer metal3 \
    -pin {rs1_sel[3]} \
    -side INSIDE
editPin \
    -assign {13.852500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[3]} \
    -side INSIDE
editPin \
    -assign {13.992500 0} \
    -layer metal3 \
    -pin {rs2_sel[3]} \
    -side INSIDE


editPin \
    -assign {14.395000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[4]} \
    -side INSIDE
editPin \
    -assign {14.647500 0} \
    -layer metal3 \
    -pin {rd_sel[4]} \
    -side INSIDE
editPin \
    -assign {15.160000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[4]} \
    -side INSIDE
editPin \
    -assign {15.300000 0} \
    -layer metal3 \
    -pin {rs1_sel[4]} \
    -side INSIDE
editPin \
    -assign {15.580000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[4]} \
    -side INSIDE
editPin \
    -assign {15.720000 0} \
    -layer metal3 \
    -pin {rs2_sel[4]} \
    -side INSIDE


editPin \
    -assign {16.122500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[5]} \
    -side INSIDE
editPin \
    -assign {16.375000 0} \
    -layer metal3 \
    -pin {rd_sel[5]} \
    -side INSIDE
editPin \
    -assign {16.887500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[5]} \
    -side INSIDE
editPin \
    -assign {17.027500 0} \
    -layer metal3 \
    -pin {rs1_sel[5]} \
    -side INSIDE
editPin \
    -assign {17.307500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[5]} \
    -side INSIDE
editPin \
    -assign {17.447500 0} \
    -layer metal3 \
    -pin {rs2_sel[5]} \
    -side INSIDE


editPin \
    -assign {17.850000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[6]} \
    -side INSIDE
editPin \
    -assign {18.102500 0} \
    -layer metal3 \
    -pin {rd_sel[6]} \
    -side INSIDE
editPin \
    -assign {18.615000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[6]} \
    -side INSIDE
editPin \
    -assign {18.755000 0} \
    -layer metal3 \
    -pin {rs1_sel[6]} \
    -side INSIDE
editPin \
    -assign {19.035000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[6]} \
    -side INSIDE
editPin \
    -assign {19.175000 0} \
    -layer metal3 \
    -pin {rs2_sel[6]} \
    -side INSIDE


editPin \
    -assign {19.577500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[7]} \
    -side INSIDE
editPin \
    -assign {19.830000 0} \
    -layer metal3 \
    -pin {rd_sel[7]} \
    -side INSIDE
editPin \
    -assign {20.342500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[7]} \
    -side INSIDE
editPin \
    -assign {20.482500 0} \
    -layer metal3 \
    -pin {rs1_sel[7]} \
    -side INSIDE
editPin \
    -assign {20.762500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[7]} \
    -side INSIDE
editPin \
    -assign {20.902500 0} \
    -layer metal3 \
    -pin {rs2_sel[7]} \
    -side INSIDE


editPin \
    -assign {21.305000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[8]} \
    -side INSIDE
editPin \
    -assign {21.557500 0} \
    -layer metal3 \
    -pin {rd_sel[8]} \
    -side INSIDE
editPin \
    -assign {22.070000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[8]} \
    -side INSIDE
editPin \
    -assign {22.210000 0} \
    -layer metal3 \
    -pin {rs1_sel[8]} \
    -side INSIDE
editPin \
    -assign {22.490000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[8]} \
    -side INSIDE
editPin \
    -assign {22.630000 0} \
    -layer metal3 \
    -pin {rs2_sel[8]} \
    -side INSIDE


editPin \
    -assign {23.032500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[9]} \
    -side INSIDE
editPin \
    -assign {23.285000 0} \
    -layer metal3 \
    -pin {rd_sel[9]} \
    -side INSIDE
editPin \
    -assign {23.797500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[9]} \
    -side INSIDE
editPin \
    -assign {23.937500 0} \
    -layer metal3 \
    -pin {rs1_sel[9]} \
    -side INSIDE
editPin \
    -assign {24.217500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[9]} \
    -side INSIDE
editPin \
    -assign {24.357500 0} \
    -layer metal3 \
    -pin {rs2_sel[9]} \
    -side INSIDE


editPin \
    -assign {24.760000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[10]} \
    -side INSIDE
editPin \
    -assign {25.012500 0} \
    -layer metal3 \
    -pin {rd_sel[10]} \
    -side INSIDE
editPin \
    -assign {25.525000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[10]} \
    -side INSIDE
editPin \
    -assign {25.665000 0} \
    -layer metal3 \
    -pin {rs1_sel[10]} \
    -side INSIDE
editPin \
    -assign {25.945000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[10]} \
    -side INSIDE
editPin \
    -assign {26.085000 0} \
    -layer metal3 \
    -pin {rs2_sel[10]} \
    -side INSIDE


editPin \
    -assign {26.487500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[11]} \
    -side INSIDE
editPin \
    -assign {26.740000 0} \
    -layer metal3 \
    -pin {rd_sel[11]} \
    -side INSIDE
editPin \
    -assign {27.252500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[11]} \
    -side INSIDE
editPin \
    -assign {27.392500 0} \
    -layer metal3 \
    -pin {rs1_sel[11]} \
    -side INSIDE
editPin \
    -assign {27.672500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[11]} \
    -side INSIDE
editPin \
    -assign {27.812500 0} \
    -layer metal3 \
    -pin {rs2_sel[11]} \
    -side INSIDE


editPin \
    -assign {28.215000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[12]} \
    -side INSIDE
editPin \
    -assign {28.467500 0} \
    -layer metal3 \
    -pin {rd_sel[12]} \
    -side INSIDE
editPin \
    -assign {28.980000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[12]} \
    -side INSIDE
editPin \
    -assign {29.120000 0} \
    -layer metal3 \
    -pin {rs1_sel[12]} \
    -side INSIDE
editPin \
    -assign {29.400000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[12]} \
    -side INSIDE
editPin \
    -assign {29.540000 0} \
    -layer metal3 \
    -pin {rs2_sel[12]} \
    -side INSIDE


editPin \
    -assign {29.942500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[13]} \
    -side INSIDE
editPin \
    -assign {30.195000 0} \
    -layer metal3 \
    -pin {rd_sel[13]} \
    -side INSIDE
editPin \
    -assign {30.707500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[13]} \
    -side INSIDE
editPin \
    -assign {30.847500 0} \
    -layer metal3 \
    -pin {rs1_sel[13]} \
    -side INSIDE
editPin \
    -assign {31.127500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[13]} \
    -side INSIDE
editPin \
    -assign {31.267500 0} \
    -layer metal3 \
    -pin {rs2_sel[13]} \
    -side INSIDE


editPin \
    -assign {31.670000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[14]} \
    -side INSIDE
editPin \
    -assign {31.922500 0} \
    -layer metal3 \
    -pin {rd_sel[14]} \
    -side INSIDE
editPin \
    -assign {32.435000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[14]} \
    -side INSIDE
editPin \
    -assign {32.575000 0} \
    -layer metal3 \
    -pin {rs1_sel[14]} \
    -side INSIDE
editPin \
    -assign {32.855000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[14]} \
    -side INSIDE
editPin \
    -assign {32.995000 0} \
    -layer metal3 \
    -pin {rs2_sel[14]} \
    -side INSIDE


editPin \
    -assign {33.397500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[15]} \
    -side INSIDE
editPin \
    -assign {33.650000 0} \
    -layer metal3 \
    -pin {rd_sel[15]} \
    -side INSIDE
editPin \
    -assign {34.162500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[15]} \
    -side INSIDE
editPin \
    -assign {34.302500 0} \
    -layer metal3 \
    -pin {rs1_sel[15]} \
    -side INSIDE
editPin \
    -assign {34.582500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[15]} \
    -side INSIDE
editPin \
    -assign {34.722500 0} \
    -layer metal3 \
    -pin {rs2_sel[15]} \
    -side INSIDE


editPin \
    -assign {35.125000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[16]} \
    -side INSIDE
editPin \
    -assign {35.377500 0} \
    -layer metal3 \
    -pin {rd_sel[16]} \
    -side INSIDE
editPin \
    -assign {35.890000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[16]} \
    -side INSIDE
editPin \
    -assign {36.030000 0} \
    -layer metal3 \
    -pin {rs1_sel[16]} \
    -side INSIDE
editPin \
    -assign {36.310000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[16]} \
    -side INSIDE
editPin \
    -assign {36.450000 0} \
    -layer metal3 \
    -pin {rs2_sel[16]} \
    -side INSIDE


editPin \
    -assign {36.852500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[17]} \
    -side INSIDE
editPin \
    -assign {37.105000 0} \
    -layer metal3 \
    -pin {rd_sel[17]} \
    -side INSIDE
editPin \
    -assign {37.617500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[17]} \
    -side INSIDE
editPin \
    -assign {37.757500 0} \
    -layer metal3 \
    -pin {rs1_sel[17]} \
    -side INSIDE
editPin \
    -assign {38.037500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[17]} \
    -side INSIDE
editPin \
    -assign {38.177500 0} \
    -layer metal3 \
    -pin {rs2_sel[17]} \
    -side INSIDE


editPin \
    -assign {38.580000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[18]} \
    -side INSIDE
editPin \
    -assign {38.832500 0} \
    -layer metal3 \
    -pin {rd_sel[18]} \
    -side INSIDE
editPin \
    -assign {39.345000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[18]} \
    -side INSIDE
editPin \
    -assign {39.485000 0} \
    -layer metal3 \
    -pin {rs1_sel[18]} \
    -side INSIDE
editPin \
    -assign {39.765000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[18]} \
    -side INSIDE
editPin \
    -assign {39.905000 0} \
    -layer metal3 \
    -pin {rs2_sel[18]} \
    -side INSIDE


editPin \
    -assign {40.307500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[19]} \
    -side INSIDE
editPin \
    -assign {40.560000 0} \
    -layer metal3 \
    -pin {rd_sel[19]} \
    -side INSIDE
editPin \
    -assign {41.072500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[19]} \
    -side INSIDE
editPin \
    -assign {41.212500 0} \
    -layer metal3 \
    -pin {rs1_sel[19]} \
    -side INSIDE
editPin \
    -assign {41.492500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[19]} \
    -side INSIDE
editPin \
    -assign {41.632500 0} \
    -layer metal3 \
    -pin {rs2_sel[19]} \
    -side INSIDE


editPin \
    -assign {42.035000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[20]} \
    -side INSIDE
editPin \
    -assign {42.287500 0} \
    -layer metal3 \
    -pin {rd_sel[20]} \
    -side INSIDE
editPin \
    -assign {42.800000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[20]} \
    -side INSIDE
editPin \
    -assign {42.940000 0} \
    -layer metal3 \
    -pin {rs1_sel[20]} \
    -side INSIDE
editPin \
    -assign {43.220000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[20]} \
    -side INSIDE
editPin \
    -assign {43.360000 0} \
    -layer metal3 \
    -pin {rs2_sel[20]} \
    -side INSIDE


editPin \
    -assign {43.762500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[21]} \
    -side INSIDE
editPin \
    -assign {44.015000 0} \
    -layer metal3 \
    -pin {rd_sel[21]} \
    -side INSIDE
editPin \
    -assign {44.527500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[21]} \
    -side INSIDE
editPin \
    -assign {44.667500 0} \
    -layer metal3 \
    -pin {rs1_sel[21]} \
    -side INSIDE
editPin \
    -assign {44.947500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[21]} \
    -side INSIDE
editPin \
    -assign {45.087500 0} \
    -layer metal3 \
    -pin {rs2_sel[21]} \
    -side INSIDE


editPin \
    -assign {45.490000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[22]} \
    -side INSIDE
editPin \
    -assign {45.742500 0} \
    -layer metal3 \
    -pin {rd_sel[22]} \
    -side INSIDE
editPin \
    -assign {46.255000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[22]} \
    -side INSIDE
editPin \
    -assign {46.395000 0} \
    -layer metal3 \
    -pin {rs1_sel[22]} \
    -side INSIDE
editPin \
    -assign {46.675000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[22]} \
    -side INSIDE
editPin \
    -assign {46.815000 0} \
    -layer metal3 \
    -pin {rs2_sel[22]} \
    -side INSIDE


editPin \
    -assign {47.217500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[23]} \
    -side INSIDE
editPin \
    -assign {47.470000 0} \
    -layer metal3 \
    -pin {rd_sel[23]} \
    -side INSIDE
editPin \
    -assign {47.982500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[23]} \
    -side INSIDE
editPin \
    -assign {48.122500 0} \
    -layer metal3 \
    -pin {rs1_sel[23]} \
    -side INSIDE
editPin \
    -assign {48.402500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[23]} \
    -side INSIDE
editPin \
    -assign {48.542500 0} \
    -layer metal3 \
    -pin {rs2_sel[23]} \
    -side INSIDE


editPin \
    -assign {48.945000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[24]} \
    -side INSIDE
editPin \
    -assign {49.197500 0} \
    -layer metal3 \
    -pin {rd_sel[24]} \
    -side INSIDE
editPin \
    -assign {49.710000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[24]} \
    -side INSIDE
editPin \
    -assign {49.850000 0} \
    -layer metal3 \
    -pin {rs1_sel[24]} \
    -side INSIDE
editPin \
    -assign {50.130000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[24]} \
    -side INSIDE
editPin \
    -assign {50.270000 0} \
    -layer metal3 \
    -pin {rs2_sel[24]} \
    -side INSIDE


editPin \
    -assign {50.672500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[25]} \
    -side INSIDE
editPin \
    -assign {50.925000 0} \
    -layer metal3 \
    -pin {rd_sel[25]} \
    -side INSIDE
editPin \
    -assign {51.437500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[25]} \
    -side INSIDE
editPin \
    -assign {51.577500 0} \
    -layer metal3 \
    -pin {rs1_sel[25]} \
    -side INSIDE
editPin \
    -assign {51.857500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[25]} \
    -side INSIDE
editPin \
    -assign {51.997500 0} \
    -layer metal3 \
    -pin {rs2_sel[25]} \
    -side INSIDE


editPin \
    -assign {52.400000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[26]} \
    -side INSIDE
editPin \
    -assign {52.652500 0} \
    -layer metal3 \
    -pin {rd_sel[26]} \
    -side INSIDE
editPin \
    -assign {53.165000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[26]} \
    -side INSIDE
editPin \
    -assign {53.305000 0} \
    -layer metal3 \
    -pin {rs1_sel[26]} \
    -side INSIDE
editPin \
    -assign {53.585000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[26]} \
    -side INSIDE
editPin \
    -assign {53.725000 0} \
    -layer metal3 \
    -pin {rs2_sel[26]} \
    -side INSIDE


editPin \
    -assign {54.127500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[27]} \
    -side INSIDE
editPin \
    -assign {54.380000 0} \
    -layer metal3 \
    -pin {rd_sel[27]} \
    -side INSIDE
editPin \
    -assign {54.892500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[27]} \
    -side INSIDE
editPin \
    -assign {55.032500 0} \
    -layer metal3 \
    -pin {rs1_sel[27]} \
    -side INSIDE
editPin \
    -assign {55.312500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[27]} \
    -side INSIDE
editPin \
    -assign {55.452500 0} \
    -layer metal3 \
    -pin {rs2_sel[27]} \
    -side INSIDE


editPin \
    -assign {55.855000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[28]} \
    -side INSIDE
editPin \
    -assign {56.107500 0} \
    -layer metal3 \
    -pin {rd_sel[28]} \
    -side INSIDE
editPin \
    -assign {56.620000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[28]} \
    -side INSIDE
editPin \
    -assign {56.760000 0} \
    -layer metal3 \
    -pin {rs1_sel[28]} \
    -side INSIDE
editPin \
    -assign {57.040000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[28]} \
    -side INSIDE
editPin \
    -assign {57.180000 0} \
    -layer metal3 \
    -pin {rs2_sel[28]} \
    -side INSIDE


editPin \
    -assign {57.582500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[29]} \
    -side INSIDE
editPin \
    -assign {57.835000 0} \
    -layer metal3 \
    -pin {rd_sel[29]} \
    -side INSIDE
editPin \
    -assign {58.347500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[29]} \
    -side INSIDE
editPin \
    -assign {58.487500 0} \
    -layer metal3 \
    -pin {rs1_sel[29]} \
    -side INSIDE
editPin \
    -assign {58.767500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[29]} \
    -side INSIDE
editPin \
    -assign {58.907500 0} \
    -layer metal3 \
    -pin {rs2_sel[29]} \
    -side INSIDE


editPin \
    -assign {59.310000 0} \
    -layer metal3 \
    -pin {rd_sel_inv[30]} \
    -side INSIDE
editPin \
    -assign {59.562500 0} \
    -layer metal3 \
    -pin {rd_sel[30]} \
    -side INSIDE
editPin \
    -assign {60.075000 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[30]} \
    -side INSIDE
editPin \
    -assign {60.215000 0} \
    -layer metal3 \
    -pin {rs1_sel[30]} \
    -side INSIDE
editPin \
    -assign {60.495000 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[30]} \
    -side INSIDE
editPin \
    -assign {60.635000 0} \
    -layer metal3 \
    -pin {rs2_sel[30]} \
    -side INSIDE


editPin \
    -assign {61.037500 0} \
    -layer metal3 \
    -pin {rd_sel_inv[31]} \
    -side INSIDE
editPin \
    -assign {61.290000 0} \
    -layer metal3 \
    -pin {rd_sel[31]} \
    -side INSIDE
editPin \
    -assign {61.802500 0} \
    -layer metal3 \
    -pin {rs1_sel_inv[31]} \
    -side INSIDE
editPin \
    -assign {61.942500 0} \
    -layer metal3 \
    -pin {rs1_sel[31]} \
    -side INSIDE
editPin \
    -assign {62.222500 0} \
    -layer metal3 \
    -pin {rs2_sel_inv[31]} \
    -side INSIDE
editPin \
    -assign {62.362500 0} \
    -layer metal3 \
    -pin {rs2_sel[31]} \
    -side INSIDE



# TODO uncomment the two below command to do pnr. These steps takes innovus more time.

# place all the standard cells in your design. This command is actually a series of many
# mini commands and settings, but it tries to optimally place the standard cells in your design
# considering area, timing, routing congestion, routing length, and other things.
# See "man place_design" to find out more.
place_design

routeDesign

# connectGlobalNets

# TODO find the command that checks DRC
verify_drc

# save your design as a GDSII, which you can open in Virtuoso
streamOut innovus.gdsii -mapFile "/class/ece425/innovus.map"

# save the design, so innovus can open it later
saveDesign $design_toplevel
