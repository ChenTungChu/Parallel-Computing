
# Set the output file type
set terminal postscript eps enhanced color solid colortext 18
# Set the output file name
set output 'mm_tile_plots.eps'

# Now plot the data with lines and points
set title "tiled matrix-matrix mult"
set title font "Helvetica,18"
set xlabel "matrix size"
set ylabel "time"
set grid
# put legend in the upper left corner
set key right top
N = 4
plot for [i=2:N] "tile.csv" u (column(1)):((column(i))) w lp\
     lw 3 title "block".(i)
#plot 'tile.csv' using 1:2 w lp lw 3 title 'length1', \
#     '' using 1:3 w lp lw 3 title 'length2', \
#     '' using 1:4 w lp lw 3 title 'length3'
