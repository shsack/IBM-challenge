
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
![Banner_site](https://user-images.githubusercontent.com/45107198/165936349-0fd0f1d0-8cf0-4005-8b25-d64f819ed087.png)


# IBM Open Science Prize 2021

The task of the [IBM Open Science Prize 2021](https://ibmquantumawards.bemyapp.com/#/event) was to simulate a Heisenberg model Hamiltonian for three interacting atoms on IBM Quantum's 7-qubit Jakarta system. The goal was to simulate the evolution of a known quantum state with the highest possbile fidelity using Trotterization. 


# Our Solution

## Reduction of the number of CNOTs

As a first step, we reduced the number of CNOTs required per Trotter step, using an optimal decomposition of the XX+YY+ZZ rotation gate that requires only 3 CNOTs.

![Screenshot 2022-04-29 at 14 13 21](https://user-images.githubusercontent.com/45107198/165942166-796a8a41-9437-40ab-8871-7ff9302237ae.png)

## Circuit compression using the Yang-Baxter Equivalence (YBE)

Next, we use the Yang-Baxter Equivalence to represent the 4 Trotter steps with an equivalent circuit that requires only 15 CNOTs.

![Screenshot 2022-04-29 at 14 14 33](https://user-images.githubusercontent.com/45107198/165942319-917540b3-1762-48c0-86fa-45aa0332dff8.png)

## Projected Variational Quantum Dynamics (pVQD)

We build upon the 4 Trotter steps circuit by variationally compressing higher order steps into the same circuit.

## Error mitigation

Finally, we use three different techniques to mitigate errors. Namely, we use an optimal qubit routing combined with Zero Noise Extrapolation (ZNE) and removal of unphysical outputs.


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/StefanoBarison"><img src="https://avatars.githubusercontent.com/u/56699595?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Stefano Barison</b></sub></a><br /><a href="https://github.com/shsack/IBM-challenge/commits?author=StefanoBarison" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/shsack"><img src="https://avatars.githubusercontent.com/u/45107198?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Stefan Hermann Sack</b></sub></a><br /><a href="https://github.com/shsack/IBM-challenge/commits?author=shsack" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
