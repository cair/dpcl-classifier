# Dynamic Probabilistic Inclusion of Literals for Concept Learning (DPCL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.13.0-brightgreen.svg)](https://www.tensorflow.org/)

DPCL is an innovative Tsetlin Machine scheme that learns concepts through propositional formulas. It's efficient in various applications and demonstrates effectiveness compared with state-of-the-art classifiers.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

Tsetlin Machine (TM) is a recent intriguing machine learning tool that learns concepts through propositional formulas. This repository contains the implementation of Dynamic Probabilistic inclusion of literals for Concept Learning (DPCL), a new Tsetlin Machine scheme with dedicated feedback tables and dynamic clause-dependent inclusion/exclusion probabilities.

## Installation

### Prerequisites

- Python 3.11 or higher
- Poetry package manager

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/cair/dpcl-classifier.git
    ```

2. Navigate to the project directory:

   ```bash
   cd dpcl-classifier
   ```

3. Install the dependencies using Poetry:

   ```bash
   poetry install
   ```

## Usage

You can run the main script as follows:
    ```bash
    python classifier_numba.py
    ```

## License 

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
