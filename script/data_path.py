from enum import Enum
from pathlib import Path


class DataPath(Enum):
    Root = Path(__file__).parents[1]
    Input = Root / 'input'
    Submission = Root / 'submissions'

    TrainCsv = Input / 'train.csv'
    TestCsv = Input / 'test.csv'

    SubmissionCsv = Input / 'sample_submission.csv'


def main():
    print(DataPath.Root)
    print(DataPath.Root.value)


if __name__ == '__main__':
    main()
