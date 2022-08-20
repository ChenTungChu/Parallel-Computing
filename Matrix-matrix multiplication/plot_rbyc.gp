
# Set the output file type to eps
# for pdf file google "gnuplot pdf output"
set terminal postscript eps enhanced color solid colortext 18
# Set the output file name
set output 'mm_rbyc_plots.eps'

# Now plot the data with lines and points
set title "row-by-column mm mult"
set title font "Helvetica,18"
set xlabel "log of matrix size"
set ylabel "time"
set grid
plot "rbyc.csv" u 1:2 w lp lw 3 title "rbyc"

