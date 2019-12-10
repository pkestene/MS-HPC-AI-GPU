# Introduction

## Historique

- rappel calcul parallèle / devinette de cuisine / notion de concurrence et parallélisme
- quand sont apparus les premiers processeurs graphiques (GPU) ? Pourquoi ?
- évolution de l'utilisation des GPU: des applications graphiques (jeux video, visualisation) vers le calcul scientifique généraliste (GPGPU : General Purpose GPU computing)
- les GPU et le calcul haute performance; quels sont les domaines d'application ?
- introduction de l'architecture CUDA en 2007

## Notions de parallélisme utiles pour comprendre les GPU

- thread, multithreading
- parallelism versus concurrency

- quelles sont les différences majeures en terme d'architecture matérielle entre un CPU et un GPU ?
- que signifie le terme "latency-oriented architecture" ?
- que signifie le terme "throughput-oriented architecture" ?
- qu'est ce que la loi d'Amdhal ?
- qu'est ce que le roofline model ? quelles sont les métriques de caractérisations de performances ?
- qu'est ce que la bande passante mémoire d'un processeur ? Comment la calcule-t-on ? la mesure-t-on ?


## bibliographie complémentaire

# Le modèle de programmation CUDA

Que signifie l'acronyme CUDA ? Que désigne-t'il ?
- un language de programmation ?
- un compilateur ?
- une architecture matérielle ?

## Architecture matérielle des GPU

- qu'est qu'un "streaming processor" ? 
- sur quels éléments repose la puissance de calcul des GPU ? 
- comment sont séquencés les threads GPU ?
- que signifie l'acronyme SIMT ?

## Architecture logicielle CUDA

- modèle PTX : Parallel Thread Execution
- qu'est ce qu'un "warp" ?
- qu'est qu'un noyau de calcul CUDA ?
- modèle hiérarchique de grille de blocs de threads; comment paramétrer l'exécution d'un noyau CUDA ? Quel lien y a-t'il avec l'architecture matérielle ? 

## Notion de modèle mémoire

- comment gérer la mémoire utilisée dans un programme CUDA (allocation, transfert synchrone/asynchrone, libération de la mémoire) ?
- qu'est ce qu'un registre ?
- qu'est ce que la mémoire partagée ?
- qu'est ce que la mémoire globale ?
- qu'est ce que la mémoire constante ?
- quels sont les goulots d'étrangelement ?
- quels type de mémoire pour quelle utilisation ?

## Le language C++/CUDA

- terminologie Host/Device
- modèle de programmation hétérogène CUDA : un même fichier source pour exécuter du code sur CPU et GPU
- quels sont les mots clés supplémentaires ajoutés par CUDA au C++ ?
- à quoi servent-ils ?
- quels sont les paramétres d'exécution d'un noyau ? que sont les variables intrinsèques ?

- quelles sont les fonctionnalités de l'interface de programmation (API) CUDA runtime ?

## Visite guidée et commentée de programmes CUDA

# Notions avancées

## pattern de programmation

- boucle for, 
- boucle de réduction, 
 -boucle scan, ...

## Analyse des limitations de performance

- outils d'aide à la validation / analyse:
  * cuda-memcheck : recherche de "race condition"
  * cuda-gdb : debogage
  * nvprof : profiling
- comment déterminer si on utilise efficacement les ressources d'un GPU ?
- coalescence mémoire
- divergence de branche

# Considérations pratiques

- installation des outils CUDA en environement linux
- compilation des exemples Nvidia
- outil d'aide à la compilation (build system) : cmake

# Travaux pratiques

- prise en main des plateformes
 * Amazon Web Service EC2
 * ROMEOLAB

- mise en oeuvre de CUDA sur 
 * des exemples simples: noyaux BLAS
 * multiplication de matrices pleines ou transposition de matrice 
 * utilisation de bibliothèques embarquant la parallélisation GPU: cuBLAS, cuFFT, ...
 * la résolution de l'équation de la chaleur sur maillage cartésien par une méthode explicite en temps

# Compléments

- autres modèles de programmation / alternatives à plus haut-niveau d'abstraction que CUDA (i.e. le code CUDA est généré par le compilateur ou par une bibliothèque): 
 * approche par directive (compilateur) : OpenMP / OpenAcc
 * approche par bibliothèque : Kokkos / Nvidia Thrust : https://fynv.github.io/ThrustRTC/Demo.html
 * approche par language de script : python (Numba/Cupy)

- quels sont les avantages d'une approche haut-niveau ? portabilité / productivité
