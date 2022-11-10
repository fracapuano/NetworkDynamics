import argparse
from HM_01.Ex_01 import execute as h1e1
from HM_01.Ex_02 import execute as h1e2
from HM_01.Ex_03 import execute as h1e3

HOMEWORK_STRUCTURE = {
    1: [(1, h1e1), (2, h1e2), (3, h1e3)]
}


def validate_homework_exercise(homework_number, exercise_number):
    exercises = HOMEWORK_STRUCTURE.get(homework_number, None)
    if exercises is None:
        return False
    if exercise_number not in exercises[0]:
        return False
    return True


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

    # Execution of homework
    HOMEWORK_STRUCTURE[args.homework][args.exercise - 1][1]()


if __name__ == "__main__":
    main()
