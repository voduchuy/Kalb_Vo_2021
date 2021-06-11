import mpi4py.MPI as MPI
import numpy as np

WORLD = MPI.COMM_WORLD
MY_RANK = WORLD.Get_rank()
NUM_WORKERS = WORLD.Get_size() - 1

NUM_EVALS = 10

if __name__ == '__main__':
    if MY_RANK == 0:
        dvs = np.linspace(0, 20, NUM_EVALS * 2)
        res = np.zeros((NUM_EVALS,))

        report_buf = np.zeros((1,), dtype=int)
        command_buf = np.zeros((1,), dtype=int)
        dv_buf = np.zeros((2,), dtype=float)
        eval_buf = np.zeros((1,), dtype=float)

        report_requests = []
        command_requests = []
        dv_send_requests = []
        eval_recv_requests = []
        for i in range(NUM_WORKERS):
            report_requests.append(WORLD.Recv_init([report_buf, 1, MPI.INT], i+1))
            command_requests.append(WORLD.Send_init([command_buf, 1, MPI.INT], i+1))
            dv_send_requests.append(WORLD.Send_init([dv_buf, 2, MPI.DOUBLE], i+1))
            eval_recv_requests.append(WORLD.Recv_init([eval_buf, 1, MPI.DOUBLE], i+1))

        i = 0
        for i in range(NUM_WORKERS):
            report_requests[i].Start()

        recv_address = [0]*NUM_WORKERS
        k = 0
        stat = MPI.Status()
        while k < NUM_EVALS:
            # Check for available CPUs
            MPI.Prequest.Waitany(report_requests, stat)
            source_rank = stat.Get_source()
            if source_rank > 0:
                print(f"CPU {source_rank} available.")
                recv_address[source_rank - 1] = k
                dv_buf[:] = dvs[2*k:2*k+2]
                command_requests[source_rank-1].Start()
                dv_send_requests[source_rank-1].Start()
                eval_recv_requests[source_rank - 1].Start()
                k+=1
            # Check for returning results
            MPI.Prequest.Waitany(eval_recv_requests, stat)
            source_rank = stat.Get_source()
            if source_rank > 0:
                print(f"CPU {source_rank} sent back result.")
                res[recv_address[source_rank - 1]] = eval_buf[0]
                report_requests[source_rank-1].Start()

        command_buf[0] = -1
        while True:
            MPI.Prequest.Waitany(eval_recv_requests, stat)
            source_rank = stat.Get_source()
            if source_rank > 0:
                res[recv_address[source_rank - 1]] = eval_buf[0]
            else:
                print("All finished.")
                break

        for request in command_requests:
            request.Start()

        print(res - dvs[0::2] - dvs[1::2])
    else:
        report_buf = np.zeros((1,), dtype=np.int32)
        root_buf = np.zeros((1,), dtype=np.int32)
        input_buf = np.zeros((2,), dtype=float)
        return_buf = np.zeros((1,), dtype=float)

        while True:
            request = WORLD.Isend([report_buf, 1, MPI.INT], 0)
            request.Wait()
            request = WORLD.Irecv([root_buf, 1, MPI.INT], 0)
            request.Wait()

            if root_buf[0] == -1:
                print(f"Process {MY_RANK} received quit signal.")
                break
            else:
                WORLD.Irecv([input_buf, len(input_buf), MPI.DOUBLE], 0)
                print(f"Process {MY_RANK} received work signal with input {input_buf}.")
                return_buf[0] = input_buf[0]+input_buf[1]
                request = WORLD.Isend([return_buf, 1, MPI.DOUBLE], 0)
                request.Wait()

