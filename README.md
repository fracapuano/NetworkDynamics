# Network Dynamics Homeworks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The following repository reflects the work done by the authors, in order to solve the homeworks assigned during the Network Dynamics course at Politecnico di Torino.

-------------------------------------------------------------

## Enviroment
Once the repo is cloned, some python libraries are required to properly setup your (virtual) enviroment.


They can be installed via pip:
```bash
    pip install -r requirements.txt
```

or via conda:
```bash
    conda create --name <env_name> --file requirements.txt
```

-------------------------------------------------------------

## Content

The `requirements.txt` includes mandatory libraries to run properly the env. 

In each homework directory, there are:
* several `.py` files, corresponding to the exercises requested in the requirements
* `utils` folder contains supporting files


-------------------------------------------------------------

## Execution

The `main.py` is the entry point of the execution.<br>
To see the help, please execute the command `python main.py -h`
. This is how to use the program:
```bash
usage: main.py [-h] -o HOMEWORK -e EXERCISE

options:
  -h, --help            show this help message and exit
  -o HOMEWORK, --homework HOMEWORK
                        Homework number
  -e EXERCISE, --exercise EXERCISE
                        Exercise number
```
Two parameters are mandatory:
* `-o --homerwork` the number of homework you want to execute
* `-e --exercise` the number of exercise inside the homework you want to execute

Then, you can run the program in this way:
```bash
python main.py --homework number_of_homework --exercise number_of_exercise 
```

-------------------------------------------------------------

## Contacts

| Author | Student Id | GitHub | 
| ------ | ------------- | ------ |
| **Lorenzo Bergadano** | s304415   | [lolloberga](https://github.com/lolloberga) |
| **Francesco Capuano** | s295366   | [fracapuano](https://github.com/fracapuano) |
| **Matteo Matteotti**  | s294552   | [mttmtt31](https://github.com/mttmtt31) |
| **Enrico Porcelli**   | s296649   | [enricoporcelli](https://github.com/enricoporcelli) |
| **Paolo Rizzo**       | s301961   | [polrizzo](https://github.com/polrizzo) |
