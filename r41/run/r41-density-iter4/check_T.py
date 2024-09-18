import json

jobs = [
    "708c9b4d7def431221ba99cdc5076a9b",
    "d6afd890c9fe5bd60b76d4879f26be7f",
    "daad153c0e8e470f0748d702a287f716",
    "8dcdd738b09bffae71da5fb3ec7af7ad",
    "3186879352ed1cbdca944746fe84e9db",
    "a3186dae8b2766353c566f5f5f7c1d31",
    "caad3d53d32a950787204e2e83fde204",
    "b9b57b610a9a04f9590325409cb292be",
    "e5a7db47943c5a54df4d3286d9ce6135",
    "ebfecc225ea95d25b93e44f9d6a80cf6",
    "46cd1f1fce4dddd63b873c29b152b2f6",
    "5011154462da3a1beb2d32378cb7f462",
    "8e6fe78694f3191d2f0358678a08904a",
    "161d0e2aec82c9a9dbceafcb70aa37fb",
    "01c76c420051d51ff5b9d551aa49417f",
    "7bdc46c85458d52dd3b4a49cb05f2759",
    "fd34e97a3fa721ee7db67080f681e19f",
    "a1edcf3fef38fd36505405d4ce0a89ce",
    "0e939e13ac47b64a03cf2f35f5455e47",
    "0a3dae6934ff982d842b03357a3914a6",
    "e1c196422943ddb2c2e843e48095786d",
    "624dae5e6d3c650cc62730f901ae0343",
    "36b30abdd271579d7c07c492eb14661d"
]

#loop through jobs in workspace
for job in jobs:
    with open("workspace/" + job + '/signac_statepoint.json', 'r') as file:
        data = json.load(file)
        T = data.get("T")
        print(f"Job ID: {job} T: {T}")
#Open .json file and look for T entry
#Print the job ide and T value