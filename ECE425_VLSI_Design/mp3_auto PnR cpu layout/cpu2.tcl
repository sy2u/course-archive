# proc is a function. this is used later to connect all the vdd! and vss! together.
proc connectGlobalNets {} {
	globalNetConnect vdd! -type pgpin -pin vdd! -all
	globalNetConnect vss! -type pgpin -pin vss! -all
	globalNetConnect vdd! -type tiehi -all
	globalNetConnect vss! -type tielo -all
	applyGlobalNets
}

# speed up compile
setMultiCpuUsage -localCpu 8

# set the top level module name (used elsewhere in the scripts)
set design_toplevel cpu

# set the verilog file to pnr
set init_verilog ../synth/outputs/$design_toplevel.v

# set the lef file of your standard cells
# when you add your regfile lef, it is here
# if you want to supply more than one lef use the following syntax:
# set init_lef_file "1.lef 2.lef"
set init_lef_file "stdcells.lef regfile.lef"

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
floorPlan -site CoreSite -s 85 50 10 10 10 10

# create the horizontal vdd! and vss! wires used by the standard cells.
sroute -allowJogging 0 -allowLayerChange 0 -crossoverViaLayerRange { metal7 metal1 } -layerChangeRange { metal7 metal1 } -nets { vss! vdd! }

# TODO for the regfile part, place the regfile marco
placeInstance datapath/bitslices[0].bitslice/regfile 0 49.92 -fixed
placeInstance datapath/bitslices[1].bitslice/regfile 0 48.64 MY -fixed
placeInstance datapath/bitslices[2].bitslice/regfile 0 47.36 -fixed
placeInstance datapath/bitslices[3].bitslice/regfile 0 46.08 MY -fixed
placeInstance datapath/bitslices[4].bitslice/regfile 0 44.80 -fixed
placeInstance datapath/bitslices[5].bitslice/regfile 0 43.52 MY -fixed
placeInstance datapath/bitslices[6].bitslice/regfile 0 42.24 -fixed
placeInstance datapath/bitslices[7].bitslice/regfile 0 40.96 MY -fixed
placeInstance datapath/bitslices[8].bitslice/regfile 0 39.68 -fixed
placeInstance datapath/bitslices[9].bitslice/regfile 0 38.40 MY -fixed
placeInstance datapath/bitslices[10].bitslice/regfile 0 37.12 -fixed
placeInstance datapath/bitslices[11].bitslice/regfile 0 35.84 MY -fixed
placeInstance datapath/bitslices[12].bitslice/regfile 0 34.56 -fixed
placeInstance datapath/bitslices[13].bitslice/regfile 0 33.28 MY -fixed
placeInstance datapath/bitslices[14].bitslice/regfile 0 32.00 -fixed
placeInstance datapath/bitslices[15].bitslice/regfile 0 30.72 MY -fixed
placeInstance datapath/bitslices[16].bitslice/regfile 0 29.44 -fixed
placeInstance datapath/bitslices[17].bitslice/regfile 0 28.16 MY -fixed
placeInstance datapath/bitslices[18].bitslice/regfile 0 26.88 -fixed
placeInstance datapath/bitslices[19].bitslice/regfile 0 25.60 MY -fixed
placeInstance datapath/bitslices[20].bitslice/regfile 0 24.32 -fixed
placeInstance datapath/bitslices[21].bitslice/regfile 0 23.04 MY -fixed
placeInstance datapath/bitslices[22].bitslice/regfile 0 21.76 -fixed
placeInstance datapath/bitslices[23].bitslice/regfile 0 20.48 MY -fixed
placeInstance datapath/bitslices[24].bitslice/regfile 0 19.20 -fixed
placeInstance datapath/bitslices[25].bitslice/regfile 0 17.92 MY -fixed
placeInstance datapath/bitslices[26].bitslice/regfile 0 16.64 -fixed
placeInstance datapath/bitslices[27].bitslice/regfile 0 15.36 MY -fixed
placeInstance datapath/bitslices[28].bitslice/regfile 0 14.08 -fixed
placeInstance datapath/bitslices[29].bitslice/regfile 0 12.80 MY -fixed
placeInstance datapath/bitslices[30].bitslice/regfile 0 11.52 -fixed
placeInstance datapath/bitslices[31].bitslice/regfile 0 10.24 MY -fixed

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
	-set_to_set_distance 4 \
	-area_blockage {0 50 70 0}
addStripe  \
	-direction vertical \
	-layer metal8 \
	-nets { vss! vdd! } \
	-spacing 1.6 \
	-width 0.4 \
	-set_to_set_distance 4 \
	-area_blockage {0 50 70 0}

# TODO restrict routing to only metal 6
setDesignMode -topRoutingLayer metal6


# TODO uncomment the two below command to do pnr. These steps takes innovus more time.

# place all the standard cells in your design. This command is actually a series of many
# mini commands and settings, but it tries to optimally place the standard cells in your design
# considering area, timing, routing congestion, routing length, and other things.
# See "man place_design" to find out more.
place_design

routeDesign

connectGlobalNets

# TODO find the command that checks DRC
verify_drc

# # save your design as a GDSII, which you can open in Virtuoso
streamOut innovus.gdsii -mapFile "/class/ece425/innovus.map"

# # save the design, so innovus can open it later
saveDesign $design_toplevel
