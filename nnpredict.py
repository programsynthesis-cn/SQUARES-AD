import argparse
import json
import torch
import numpy as np
# torch.cuda.current_device()
import torch.multiprocessing as mp
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import os
os.environ['R_HOME'] = 'C:\\Users\\sdtbu\\.conda\\envs\\squares\\Lib\\R'
os.environ['R_USER'] = 'C:\\Users\\sdtbu\\.conda\\envs\\squares\\Lib\\site-packages\\rpy2'
from baseline.robustfill.model2 import RobustFill
from train import load_data
from env.env import ProgramEnv
from dsl.example import Example
from dsl.program import Program

import sys
sys.path.append("..")

from squaresEnumerator import Squares
import time

txtfile = open('D:\\6.working\SQUARES-master\out.txt','a')
input_file_name = ""
# def input_file():
#     input_file_name = "D:\\6.working\SQUARES-master\\tests-examples\\55-tests//1.in"
#     return input_file_name
def main():
    # spec_str()

    s = Squares()
    inputs = []

    consts, aggrs, attrs, loc = "", "", "", 1
    output = ""
    start_time = time.time()
    # print(spec_str)

    global kg
    kg = 1
    dir = "../"
    input0 = "id,pname\n1,Tire\n2,Suspension"
    input1 = "id,sname\n1,Michelin"
    output = "pname\nTire"
    consts = ""
    aggrs = ""
    attrs = ""
    inputs.append(input0)
    inputs.append(input1)
    # global input_file_name0101

    input_file_name = "D:\\6.working\SQUARES-master\\tests-examples\\55-tests/1.in"
    # input_file_name = "D:\\6.working\SQUARES-master\\tests-examples\\55-tests//46.in"
    # input_file()
    print(input_file_name, file=txtfile)
    # R_Query, SQL_Query = s.synthesize(inputs, output, consts, aggrs, attrs, loc,input_file_name)
    R_Query = s.synthesize(inputs, output, consts, aggrs, attrs, loc,input_file_name)
    # print(SQL_Query)
    # print(SQL_Query,file=txtfile)
    print(R_Query)
    print(R_Query,file=txtfile)
    Time = time.time() - start_time
    print("time = ", Time,file=txtfile)  # 0.46~0.48

    print("time = ", Time)  # 0.46~0.48
    txtfile.close()


if __name__ == '__main__':
    main()




# def init_worker(*args):
#     global counter, fail_counter, model, program_len, timeout
#     counter, fail_counter, model, program_len, timeout = args
#
#
# def robustfill_cab(env, max_depth, model, beam_size, width, timeout, input, input_lens, output,
#         output_lens, input_masks, output_masks):
#
#     start_time = time.time()
#     state = {'num_steps': 0, 'end_time': start_time + timeout}
#
#     res = False
#     print('29 ',time.time())
#     while time.time() < state['end_time']:
#         #2021.8.16
#
#         res = model.beam_search(env, max_depth, input, input_lens, output, output_lens, input_masks,
#                                 output_masks, beam_size, width, state)
#
#         return res
#     #     if res is not None:
#     #         break
#     #     beam_size *= 2
#     #
#     # ret = {'result': res, 'num_steps': state['num_steps'], 'time': time.time() - start_time,
#     #        'beam_size': beam_size, 'width': width}
#     # return ret
#
#
# def solve_problem_worker(args):
#     line, input, input_lens, output, output_lens, input_masks, output_masks = args
#     examples = Example.from_line(line)
#     sol = Program.parse(line['program'])
#     env = ProgramEnv(examples)
#     res = robustfill_cab(env, program_len, model, 100, 48, timeout, input, input_lens, output, output_lens,
#                          input_masks, output_masks)
#
#     counter.value += 1
#     print("\rSolving problems... %d (failed: %d)" % (counter.value, fail_counter.value), end="")
#
#     if res['result'] is None:
#         res['result'] = "Failed"
#         fail_counter.value += 1
#         return res
#     else:
#         res['result'] = str(Program(sol.input_types, res['result']))
#         return res
#
#
# def solve_problems(input_path, program_len, model, num_workers, timeout):
#     # Prevents deadlocks due to torch's problems with GPUs on multi processes.
#     # This line is here for convenience, but it is recommended to solve problems on CPU since the overhead
#     # in this case is minimal.
#     torch.set_num_threads(1)
#     db_columns = []
#     f_in = open(input_path, 'r')
#     inputs = f_in.readline()[:-1].split(":")[1].replace(" ","").split(",")
#     for i in inputs:
#         with open(i, 'r') as f:
#             db_columns = list(set(db_columns +f.readlines()))
#
#     db_columns = ','.join(db_columns)
#     db_columns = db_columns.replace("\n", ",")
#     # db_columns = '"'+db_columns+'"'
#     # db_columns = list(db_columns)
#     print(db_columns)
#
#     output = f_in.readline()[:-1].split(":")[1].replace(" ","")
#     with open(output, 'r') as f:
#         cols = f.readline()
#     cols = ''.join(cols)
#     cols = cols.replace("\n", ",")
#     # cols = '"'+cols+'"'
#     print(cols)
#
#     # {"program":"Empty|inner_join|inner_join3|inner_join4","examples":[{"inputs":[["C_name,F_key\nclass1,f1"]],"output":["Sname"]}]}
#     # outstr =
#     # inputstr =
#     # strtest = "{"+'"'+"program"+'"'+":"+'"'+'"'+","+'"'+"examples"+'"'+":"+"["+"{"+'"'+"inputs"+'"'+":"+"["+"["+db_columns+"]"+"]"+","+'"'+"output"+'"'+":"+"["+cols+"]"+"}"+"]"+"}"
#     strtest = "{\"program\":\"\",\"examples\":[{\"inputs\":[[\""+db_columns+"\"]],\"output\":[\""+cols+"\"]}]}"
#     lines = []
#     lines.append(strtest)
#     # lines[0].replace("'","")
#     print(lines)
#     # with open(input_path, 'r') as f:
#     #     #list ['{"program":"Empty|inner_join|select","examples":
#     #     # [{"inputs":[["C_name,F_key\\nclass1,f1\\nclass2,f2\\nclass3,f1\\n\\nclass4,f3\\nclass5,f4",
#     #     # "S_key,C_name\\nS1,class1\\nS2,class1\\nS3,class2\\nS3,class5\\nS4,class2\\nS4,class4\\nS5,class3\\nS6,class3\\nS6,class2\\nS7,class5\\nS8,class4",
#     #     # "F_key,F_name\\nf1,faculty1\\nf2,faculty2\\nf3,faculty3\\nf4,faculty4",
#     #     # "S_key,S_name,level\\nS1,stu1,JR\\nS2,stu2,SR\\nS3,stu3,JR\\nS4,stu4,SR\\nS5,stu5,JR\\nS6,stu6,SR\\nS7,stu7,JR\\nS8,stu8,JR"]]
#     #     # ,"output":["S_name\\nstu1\\nstu5"]}]}']
#     #     lines = f.read().splitlines()
#     data = load_data(lines, num_workers)
#     filtered_data = dict()
#
#     for k in ['input', 'input_lens', 'output', 'output_lens']:
#         filtered_data[k] = torch.LongTensor(data[k])
#
#     for k in ['input_padding_mask', 'output_padding_mask']:
#         filtered_data[k] = torch.FloatTensor(data[k])
#
#     print('78 ',filtered_data)
#
#     worker_data = []
#     for i, line in enumerate(lines):
#         line_data = {}
#         for k, v in filtered_data.items():
#             line_data[k] = v[i].unsqueeze(0)
#             # line_data[k] = v[i].cuda()
#
#         worker_data.append((json.loads(lines[i]), line_data['input'], line_data['input_lens'],
#                             line_data['output'], line_data['output_lens'], line_data['input_padding_mask'],
#                             line_data['output_padding_mask']))
#     print('89 ',worker_data)
#
#     # lines = f.read().splitlines()
#
#     counter = mp.Value('i', 0)
#     fail_counter = mp.Value('i', 0)
#
#
#
#
#     # if num_workers is None or num_workers > 1:
#     #     pool = mp.Pool(processes=num_workers, initializer=init_worker,
#     #                                 initargs=(counter, fail_counter, model, program_len, timeout))
#     #     # return pool.map(solve_problem_worker, worker_data)
#     #     return pool.map(solve_problem_worker, [worker_data])
#     #     # args = worker_data
#     #     # return solve_problem_worker(args)
#     #     # return 0
#     # else:
#     #     # Don't run in pool to enable debugging
#     #     init_worker(counter, fail_counter, model, program_len, timeout)
#     #     return [solve_problem_worker(data) for data in worker_data]
#
#     line = json.loads(lines[i])
#     input = line_data['input']
#     input_lens = line_data['input_lens']
#     output = line_data['output']
#     output_lens = line_data['output_lens']
#     input_masks = line_data['input_padding_mask'],
#     output_masks = line_data['output_padding_mask']
#     examples = Example.from_line(line)
#     sol = Program.parse(line['program'])
#     env = ProgramEnv(examples)
#     res = robustfill_cab(env, program_len, model, 100, 48, timeout, input, input_lens, output, output_lens,
#                          input_masks, output_masks)
#
#     print('135 ',res)
#     return res
#     # counter.value += 1
#     # print("\rSolving problems... %d (failed: %d)" % (counter.value, fail_counter.value), end="")
#     #
#     # if res['result'] is None:
#     #     res['result'] = "Failed"
#     #     fail_counter.value += 1
#     #     return res
#     # else:
#     #     res['result'] = str(Program(sol.input_types, res['result']))
#     #     return res
# def spec_str():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--input_path', type=str,default='D:\\6.working\SQUARES-master\\test_dataset-2')
#     parser.add_argument('--input_path', type=str,default='D:\\6.working\SQUARES-master\\tests-examples\\55-tests/1.in')
#     parser.add_argument('--output_path', type=str,default='D:\\6.working\SQUARES-master\\result1')
#     parser.add_argument('--model_path', type=str,default='D:\\6.working\SQUARES-master\\nn\\model.30')
#     parser.add_argument('--timeout', type=int,default=1)
#     parser.add_argument('--max_program_len', type=int,default=5)
#     parser.add_argument('--num_workers', type=int, default=None)
#     parser.add_argument('--max_beam_size', type=int, default=6400)
#
#     # dir = "../"
#     # global input_file_name
#     # input_file_name = dir + "tests-examples\\55-tests/1.in"
#     file_path = 'D:\\6.working\SQUARES-master\example\squares3.tyrell'
#     args = parser.parse_args()
#
#     model = RobustFill()
#
#     model.load(args.model_path)
#     # model.cuda(device=0)
#     # model.cpu()
#     model.eval()
#
#     # model2 191
#     #0:'PAD',1:'<',2:'>',3:Empty,4:inner_join,5:inner_join3,6:inner_join4,7:anti_join,8:left_join,9:bind_rows,10:intersect,11:select
#     res = solve_problems(args.input_path, args.max_program_len, model, args.num_workers, args.timeout)
#     print('117 ',res)
#     print('')
#
#
#
#
#
#     # f_in = open("D:\\6.working\SQUARES-master\example\squares3.trell", 'r')
#
#     # read the list of attributes from the input file 从输入文件中读取属性列表
#     # attrs = f_in.readline()[:-1].replace(" ", "").split(":")
#     # attrs = f_in.readlines()
#
#     with open(file_path, 'r') as f:
#         spec_str1 = f.read()
#     with open(file_path, 'r') as f:
#         spec_str = f.readlines()
#
#     print(spec_str)
#
#     # 0:'PAD',1:'<',2:'>',3:Empty,4:inner_join,5:inner_join3,6:inner_join4,7:anti_join,8:left_join,9:bind_rows,10:intersect,11:select
#     dropouta = res.numpy().tolist()
#     print(dropouta)
#     a1 = dropouta[0]
#     dropout = a1[9:]
#
#     for i, j in enumerate(dropout):
#         i = int(i)
#         if dropout[i] == 3:
#             spec_str[38] = "\n"
#         elif dropout[i] == 4:
#             spec_str[40], spec_str[41], spec_str[42], spec_str[43], spec_str[88], spec_str[89], spec_str[91], spec_str[
#                 95], \
#             spec_str[100], spec_str[104] = "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n"
#         elif dropout[i] == 5:
#             spec_str[45], spec_str[46], spec_str[47], spec_str[48], spec_str[88], spec_str[89], spec_str[91], spec_str[
#                 92], \
#             spec_str[93], spec_str[94], spec_str[96], spec_str[
#                 103] = "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n"
#         elif dropout[i] == 6:
#             spec_str[50], spec_str[51], spec_str[52], spec_str[89], spec_str[93], spec_str[95], spec_str[96], spec_str[97]\
#                 , spec_str[101], spec_str[102] = "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n"
#         elif dropout[i] == 7:
#             spec_str[54], spec_str[55], spec_str[56], spec_str[57], spec_str[58], spec_str[90], spec_str[94], spec_str[
#                 98], \
#             spec_str[99], spec_str[100], spec_str[101], spec_str[
#                 105] = "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n", "\n"
#         elif dropout[i] == 8:
#             spec_str[60], spec_str[61], spec_str[62], spec_str[63] = "\n", "\n", "\n", "\n"
#         elif dropout[i] == 9:
#             spec_str[65], spec_str[66], spec_str[67], spec_str[68] = "\n", "\n", "\n", "\n"
#         elif dropout[i] == 10:
#             spec_str[70], spec_str[71], spec_str[72], spec_str[73] = "\n", "\n", "\n", "\n"
#
#     # for index, item in enumerate(spec_str):
#     #     print(index, item)
#
#     spec_str = ''.join(spec_str)
#
#     # print(spec_str1)
#     print(spec_str)
#
#     return dropout,spec_str
#
# # def filename():
# #     for i in range(1, 55):
# #         dir = "./"
# #         i = str(i)
# #         input_file_name = dir + "tests-examples\\55-tests//" + i + ".in"
# #         print(input_file_name, file=txtfile)
# #     return input_file_name
