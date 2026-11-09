# SolveIt-engineering-tool

**SolveIt** is an engineering-oriented calculation tool designed to automate repetitive mathematical operations while keeping the engineer responsible for understanding and solving the problem.

The idea behind SolveIt is simple:

> **The engineer defines the problem. SolveIt performs the calculations.**

SolveIt is not intended to replace engineering reasoning or act as a general-purpose AI problem solver. Instead, it provides computational tools that allow engineers and students to focus on the **analysis, modeling, assumptions, and interpretation** of a problem while letting the computer handle repetitive symbolic and numerical calculations.

## What is SolveIt?

SolveIt is a web-based application developed with **Python, Flask, and SymPy**. It provides an interface for performing mathematical calculations and solving engineering-related problems using a symbolic mathematics engine.

The project was created as an engineering learning project, with the goal of applying programming and mathematical knowledge to the automation of calculations commonly encountered in engineering.

### Current capabilities

SolveIt is being developed around several areas, including:

* Mathematical calculations
* Symbolic algebra
* Polynomial calculations
* Derivatives
* Equations and systems
* Engineering-oriented calculations
* Physics calculations
* Automated mathematical manipulation

The default application frontend is [`solveit_newfrontend.html`](./solveit_newfrontend.html).
It is served at the root URL (`/`) and presents six responsive sections:
Calculator, Calculus, Algebra, Physics Lab, Chemistry Lab, and Units Converter.
The Chemistry Lab adds periodic-element lookup, gas laws and the rule of three,
solution nomenclature (`st`/`sc`), equation balancing, limiting reactants,
theoretical yield, kinetics, radioactive decay, and equilibrium calculations.
These operations are exposed through the JSON `POST /api/chemistry` route.

The Units Converter is available as a sixth frontend mode. It provides two-way
editable input/output displays and converts across length, mass, time, area,
volume, speed, acceleration, force, pressure, energy, power, frequency, angle,
charge, voltage, resistance, temperature, and digital-data units through
`GET /api/units` and `POST /api/units`.

The scientific calculator defaults to degrees while retaining selectable
radians and gradians. Chemistry element results include common oxidation states,
and symbolic results render every returned LaTeX field in the frontend.

The application uses **SymPy** as its mathematical engine, allowing it to work with symbolic expressions rather than relying exclusively on numerical approximations.

## Engineering Philosophy

SolveIt follows an engineering approach to problem solving:

1. **Identify the problem**
2. **Determine what is known and unknown**
3. **Establish the appropriate model**
4. **Define the equations and constraints**
5. **Use computational tools to perform the required calculations**
6. **Analyze and verify the result**
7. **Apply the result to the original engineering problem**

The software is therefore intended to be a **tool within the engineering process**, rather than the process itself.

A computer can calculate extremely quickly, but it still depends on the user to determine **what should be calculated and why**.

## Technology

SolveIt currently uses:

* **Python** — Core programming language
* **Flask** — Web application framework
* **SymPy** — Symbolic mathematics engine
* **HTML/CSS/JavaScript** — User interface

## Project Development

SolveIt is an evolving project. New functionality, improvements to the interface, additional engineering calculations, and new problem-solving modes are planned as development continues.

The long-term objective is to develop SolveIt into a broader engineering calculation environment covering areas such as:

* Mathematics
* Physics
* Kinematics
* Dynamics
* Engineering mechanics
* Automated engineering calculations

while maintaining the original philosophy of **supporting engineering reasoning rather than replacing it**.

## Author

**Developed by Toto Bermúdez**

Industrial Engineering student and developer interested in engineering, mathematics, mechanics, programming, and computational tools.

---

**SolveIt — Calculate faster. Think like an engineer.**
