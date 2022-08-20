# Set the output file type
set terminal postscript eps enhanced color solid colortext 18
# Set the output file name
set output 'gauss_plots_pthread.eps'

# Now plot the data with lines and points
set title "pthread Gauss solver"
set title font "Helvetica,18"
set xlabel "log of matrix size"
set ylabel "time"
set grid
# put legend in the upper left corner
set key left top
N = 5
plot for [i=2:N] "Gauss_solver_pthread.csv" u (column(1)):((column(i))) w lp\
     lw 3 title "thrs-".(2**(i-2))

