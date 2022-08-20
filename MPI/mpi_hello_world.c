/* MPI hello world program that uses MPI_Get_processor_name */

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  
  int world_size, world_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

// Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("Hello from PE %s, rank %d out of %d PEs\n",
         processor_name, world_rank, world_size);

  MPI_Finalize();
  return 0;
}
