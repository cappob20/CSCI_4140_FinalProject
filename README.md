# Final Project: Option 2
<sup>Brianna Cappo (cappob20), Brandon Gammon (gammonb19), Alex Rowe (rowea20)</sup>

The goal of this project was to create a program that will predict the next 3 characters in a sentence using any language. The dataset has a maximum time limit of 30 minutes to load, and must contain multiple languages. After the user chooses the i​​th character, the system should choose the top 3 candidates for the (i ​+ 1)th character as quickly as it can.
We collected a dataset that contains the first chapter of *Harry Potter* in languages English, German, Russian and Spanish.

## Installation
Install dependencies with `pip install`:
* `keras`
* `tensorflow`

## Running
For test mode run:
`python src/runner.py test --work_dir work --test_data example/input.txt --test_output output/pred.txt`

To train the model run:
`python src/runner.py train --work_dir work`
