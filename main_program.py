import os
import getopt, sys


def main_program(program, argv):
    try:
        opts, args = getopt.getopt(argv, "i:x:n:c:", ['input_file=', 'x_offset=', 'new_file=', 'class_n='])
    except getopt.GetoptError:
        print('%s: invalid parameters' % program)
        sys.exit(1)
    file_path = "D:\churn"
    input_file = "churn_learning.txt"
    new_data_file = "churn_targeting.txt"

    # condition
    x_offset = 1
    class_n = 1
    for o, a in opts:
        if o in ('-i', '--input_file'):
            input_file = a
        elif o in ('-x', '--x_offset'):
            x_offset = int(a)
        elif o in ('-n', '--new_file'):
            new_data_file = a
        elif o in ('-c', '--class_n'):
            class_n = int(a)
    check_path = os.path.join(file_path, 'checkpoint')
    data_path = os.path.join(file_path, 'data')
    result_path = os.path.join(file_path, 'results')
    if not os.path.exists(check_path):
        os.mkdir(check_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    from utils.pipe import main
    main(input_file, new_data_file,
         data_path=data_path, check_path=check_path, result_path=result_path,
         epoch=10, batch_size=1000000, verbose=1, save=True, load_weights=False,
         class_n=1, x_offset=1, random_state=0)

if __name__ == '__main__':
    main_program(sys.argv[0], sys.argv[1:])