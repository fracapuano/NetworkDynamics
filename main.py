import argparse
from HW_01.Ex_01 import execute as h1e1
from HW_01.Ex_02 import execute as h1e2
from HW_01.Ex_03 import execute as h1e3

from HW_02.Ex_01 import execute as h2e1
from HW_02.Ex_02 import execute as h2e2
from HW_02.Ex_03 import execute as h2e3

from HW_03.Ex_01 import execute as h3e1
from HW_03.Ex_02 import execute as h3e2

HOMEWORK_STRUCTURE = {
    1: [(1, h1e1), (2, h1e2), (3, h1e3)],
    2: [(1, h2e1), (2, h2e2), (3, h2e3)],
    3: [(1, h3e1), (2, h3e2)]
}


def validate_homework_exercise(homework_number, exercise_number):
    exercises = HOMEWORK_STRUCTURE.get(homework_number, None)
    if exercises is None:
        return False
    for pair in exercises:
        if exercise_number == pair[0]:
            return True
    return False


def set_mandatory_args(parser):
    parser.add_argument("-o", "--homework", type=int, help="Homework number", required=True)
    parser.add_argument("-e", "--exercise", type=int, help="Exercise number", required=True)


def main():
    parser = argparse.ArgumentParser(description="Solutions of Network Dynamics and Learning course")
    set_mandatory_args(parser)

    args = parser.parse_args()
    print(f'Received homerwork {args.homework}, exercise {args.exercise}')
    if not validate_homework_exercise(args.homework, args.exercise):
        print('Not a valid homerwork or exercise number')
        exit(-1)

    # BUG fixing...
    if args.homework == 3 and args.exercise == 1:
        raise ValueError("Homework 3 Exercise 1 cannot be executed from here! Please check `HW3_Ex1.ipynb` in the HW_03 folder.")

    # Execution of homework
    HOMEWORK_STRUCTURE[args.homework][args.exercise - 1][1]()


if __name__ == "__main__":
    main()
