#!/usr/bin/env python3

'''
    Builds and submits a linkinvent script for de novo linker design using 
    REINVENT 4.0
    Author: A dumb fucking clanker + Jordan Harrison

    This version of the script has been slightly refactored to allow for
    programmatic generation of a SLURM submission script.  The
    ``write_slurm_script`` function now accepts a number of keyword
    arguments corresponding to the most commonly customised SLURM
    directives such as job name, output/error file names, GPU/CPU
    resources, memory and walltime.  If no ``time`` argument is given
    but a PDB file path is supplied via ``pdb_path`` the function will
    estimate a reasonable walltime based on the number of atoms in the
    structure.

    Additionally, this refactored version will attempt to locate the
    user‑installed Python library path at runtime and append it to
    ``PYTHONPATH`` in the generated SLURM script.  On clusters like
    Compute Canada users often install additional packages in their
    home directories; exporting this path ensures those modules are
    discoverable at runtime without hard‑coding absolute paths.

    Usage example::

        # generate a SLURM script with defaults
        write_slurm_script(output_toml="sampling.toml")

        # generate a script with a custom walltime derived from a PDB file
        write_slurm_script(output_toml="sampling.toml", pdb_path="complex.pdb")

        # override specific directives
        write_slurm_script(
            output_toml="sampling.toml",
            job_name="linkinvent_custom",
            mem="32G",
            cpus=8,
            time="0-02:00"
        )

    When run as a script, this module accepts command‑line arguments for
    the input SMILES, distance file and output TOML as before.  New
    optional flags allow the user to customise resource requests from
    the command line without editing the code.
'''

import argparse
import os
import subprocess
import toml
import site
from typing import Optional
from rdkit import Chem
from rdkit.Chem import rdmolops, rdMolDescriptors

def molecule_features(smiles: str) -> Optional[dict]:
    """
    Extracts molecular features from a SMILES string.
    :param smiles: Input SMILES string
    :return: Dictionary of molecular features or ``None`` if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolecularWeight": rdMolDescriptors.CalcExactMolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBondAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "HBondDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumRotBond": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "SlogP": rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
    }

def extract_warhead_smiles(smiles: str) -> tuple:
    """
    Extracts the two warhead SMILES from a PROTAC SMILES string.
    Assumes the format is ``warhead1|warhead2``.

    :param smiles: smiles string in the format "warhead1|warhead2"
    :raises ValueError: if the input string does not contain exactly one
        '|' separator
    :return: Tuple of warhead SMILES strings (warhead1, warhead2)
    """
    parts = smiles.split('|')
    if len(parts) != 2:
        raise ValueError(f"Invalid PROTAC SMILES format: {smiles}")
    return parts[0].strip(), parts[1].strip()

def longest_path_length(smiles: str) -> int:
    """Return the number of bonds in the longest path of a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    dmat = rdmolops.GetDistanceMatrix(mol)
    return int(dmat.max())

def generate_toml(smiles_csv: str, dist_file: str, output_toml: str) -> None:
    """
    Generate a staged sampling TOML configuration for REINVENT/Link‑INVENT.

    Reads the minimum/maximum distance values from ``dist_file`` and the
    first two SMILES strings from ``smiles_csv``, then uses
    ``molecule_features`` and ``longest_path_length`` to derive
    descriptor‑dependent scoring functions.  The resulting configuration is
    written to ``output_toml``.
    """
    # Extract distances
    with open(dist_file, 'r') as f:
        min_dist, max_dist = map(float, f.readline().strip().split(','))

    # Read two SMILES from CSV; expects a single line with 'smiles1|smiles2'
    with open(smiles_csv, 'r') as f:
        smiles1, smiles2 = extract_warhead_smiles(f.readline().strip())

    chem_data = [molecule_features(s) for s in (smiles1, smiles2)]
    # Fallback if parsing failed
    if any(d is None for d in chem_data):
        raise ValueError("Unable to parse one or both SMILES strings")

    weight = sum(d.get("MolecularWeight", 0) for d in chem_data)
    TPSA = sum(d.get("TPSA", 0) for d in chem_data)
    HBondAcceptors = sum(d.get("HBondAcceptors", 0) for d in chem_data)
    HBondDonors = sum(d.get("HBondDonors", 0) for d in chem_data)
    NumRotBond = sum(d.get("NumRotBond", 0) for d in chem_data)
    NumRings = sum(d.get("NumRings", 0) for d in chem_data)
    NumAromaticRings = sum(d.get("NumAromaticRings", 0) for d in chem_data)
    SlogP = sum(d.get("SlogP", 0) for d in chem_data)
    length = longest_path_length(smiles1) + longest_path_length(smiles2)

    # Construct the scoring configuration
    stages = [
        {
            "termination": "simple",
            "chkpt_file": "",  # Will be set per stage
            "max_score": 0.6,
            "min_steps": 1000,
            "max_steps": 5000,
            "scoring": {
                "type": "geometric_mean",
                "component": [
                    {
                        "MolecularWeight": {
                            "endpoint": [
                                {
                                    "name": "Molecular weight",
                                    "weight": 1,
                                    "transform": {
                                        "type": "double_sigmoid",
                                        "high": weight + int(max_dist) * 5,
                                        "low": weight + int(min_dist) * 5,
                                        "coef_div": 500.0,
                                        "coef_si": 20.0,
                                        "coef_se": 20.0,
                                    },
                                }
                            ]
                        }
                    },
                    {
                        "FragmentGraphLength": {
                            "endpoint": [
                                {
                                    "name": "Molecule length (number of bonds in longest path)",
                                    "weight": 5,
                                    "transform": {
                                        "type": "sigmoid",
                                        "high": int(max_dist) * 1.25 + 7.5,
                                        "low": int(min_dist) * 1.25 + 7.5,
                                        "k": 0.5,
                                    },
                                }
                            ]
                        }
                    },
                    {
                        "FragmentEffectiveLength": {
                            "endpoint": [
                                {
                                    "name": "Effective length (distance between anchor atoms)",
                                    "weight": 5,
                                    "transform": {
                                        "type": "sigmoid",
                                        "high": int(max_dist) + 5,
                                        "low": int(min_dist) + 5,
                                        "k": 0.5,
                                    },
                                }
                            ]
                        }
                    },
                    {
                        "FragmentLengthRatio": {
                            "endpoint": [
                                {
                                    "name": "Length ratio (effective / graph length)",
                                    "weight": 5,
                                    "transform": {
                                        "type": "sigmoid",
                                        "high": 1.0,
                                        "low": 0.99,
                                        "k": 0.5,
                                    },
                                }
                            ]
                        }
                    },
                    {
                        "TPSA": {
                            "endpoint": [
                                {
                                    "name": "TPSA",
                                    "weight": 1,
                                    "transform": {
                                        "type": "double_sigmoid",
                                        "high": TPSA + 45.0,
                                        "low": TPSA + 0,
                                        "coef_div": 140.0,
                                        "coef_si": 20.0,
                                        "coef_se": 20.0,
                                    },
                                }
                            ]
                        }
                    },
                    {
                        "FragmentHBondAcceptors": {
                            "endpoint": [
                                {
                                    "name": "Number of HB acceptors (Lipinski)",
                                    "weight": 2,
                                    "transform": {
                                        "type": "reverse_sigmoid",
                                        "high": 5,
                                        "low": 0,
                                        "k": 0.5,
                                    },
                                }
                            ]
                        }
                    },
                    {
                        "FragmentHBondDonors": {
                            "endpoint": [
                                {
                                    "name": "Number of HB donors (Lipinski)",
                                    "weight": 2,
                                    "transform": {
                                        "type": "reverse_sigmoid",
                                        "high": 5,
                                        "low": 0,
                                        "k": 0.5,
                                    },
                                }
                            ]
                        }
                    },
                ],
            },
        }
    ]
    config = {
        "linkinvent": {
            "scoring_function": {
                "name": "geometric_mean",
                "parameters": {
                    "smiles": f"{smiles1}|{smiles2}",
                    "distance": [min_dist, max_dist],
                    "weight": weight,
                    "TPSA": TPSA,
                    "HBondAcceptors": HBondAcceptors,
                    "HBondDonors": HBondDonors,
                    "NumRotBond": NumRotBond,
                    "NumRings": NumRings,
                    "NumAromaticRings": NumAromaticRings,
                    "SlogP": SlogP,
                    "length": length,
                },
            },
            "stage": stages,
        }
    }
    with open(output_toml, 'w') as f:
        toml.dump(config, f)
    print(f"Staged learning TOML configuration written to {output_toml}")

def _estimate_walltime(atom_count: int) -> str:
    """
    Estimate a reasonable SLURM walltime string based on the number of atoms.

    The base job length is 30 minutes.  For every additional 1,000 atoms
    beyond the first 1,000 an additional 15 minutes is added.  If the total
    exceeds 24 hours, the days field will be used.

    :param atom_count: total number of ATOM and HETATM entries in the PDB
    :return: walltime in the format ``D-HH:MM`` compatible with SLURM
    """
    base_minutes = 30
    extra_minutes = 0
    if atom_count > 1000:
        extra_minutes = ((atom_count - 1000) // 1000 + 1) * 15
    total_minutes = base_minutes + extra_minutes
    days = total_minutes // (24 * 60)
    hours = (total_minutes % (24 * 60)) // 60
    minutes = total_minutes % 60
    return f"{days}-{hours:02d}:{minutes:02d}"

def _count_atoms(pdb_path: str) -> int:
    """
    Count the number of ATOM and HETATM records in a PDB file.
    :param pdb_path: path to the PDB file
    :return: number of atoms counted
    """
    count = 0
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                count += 1
    return count

def write_slurm_script(
    output_toml: str,
    slurm_script: str = "submit_linkinvent.sh",
    *,
    job_name: str = "linkinvent",
    output: str = "linkinvent.out",
    error: str = "linkinvent.err",
    gres: str = "gpu:1",
    mem: str = "16G",
    cpus: int = 4,
    time: Optional[str] = None,
    account: str = "def-aminpour",
    mail_type: str = "ALL",
    mail_user: str = "jaharri1@ualberta.ca",
    pdb_path: Optional[str] = None,
    include_user_site: bool = True,
) -> None:
    """
    Write a SLURM submission script customised to the supplied resource
    requirements.

    If ``time`` is not provided and ``pdb_path`` is given, the walltime will
    be estimated using the number of atoms in the PDB file via
    :func:`_estimate_walltime`.  All other arguments directly map to their
    corresponding ``#SBATCH`` directives.
    """
    # Determine walltime
    if time is None:
        if pdb_path is not None:
            atom_count = _count_atoms(pdb_path)
            time = _estimate_walltime(atom_count)
            print(f"Estimated walltime based on {atom_count} atoms: {time}")
        else:
            time = "0-00:30"

    # Locate user site packages if requested
    user_site_line = ""
    if include_user_site:
        try:
            user_site = site.getusersitepackages()
            user_site_line = f"export PYTHONPATH=$PYTHONPATH:{user_site}\\n"
        except Exception:
            user_site_line = ""

    slurm_contents = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --gres={gres}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --mail-type={mail_type}
#SBATCH --mail-user={mail_user}

module load StdEnv/2023
module load openbabel/3.1.1
module load gcc/12.3
module load cmake
module load cuda/12.6
module load python/3.11.5
module load scipy-stack/2025a
module load rdkit/2024.09.6
module load python-build-bundle/2025b

{user_site_line}
echo "Running REINVENT Link-INVENT sampling..."
reinvent -l staged.log {output_toml}

echo "Exit code: $?"
echo "Job ran successfully"

# subsequent analyses
echo "Running Docking..."
python dock.py

echo "Running Molecular Dynamics..."
python md.py

echo "Running Analysis..."
python analysis.py
"""
    with open(slurm_script, 'w') as f:
        f.write(slurm_contents)
    print(f"SLURM script written to {slurm_script}")

def submit_job(slurm_script: str = "submit_linkinvent.sh") -> None:
    """Submit a SLURM job script using ``sbatch``."""
    print("Submitting job with sbatch…")
    subprocess.run(["sbatch", slurm_script])
    print("Job submitted.")

def main() -> None:
    """
    Entry point for the script when run from the command line.

    Supports the original arguments for generating the sampling TOML
    configuration plus new optional arguments to customise the SLURM
    submission.  If a PDB file is supplied and the walltime is not,
    the script will automatically estimate the walltime using the atom
    count.
    """
    parser = argparse.ArgumentParser(
        description="Build and submit a Link-INVENT REINVENT sampling job using TOML + SLURM."
    )
    parser.add_argument("--smiles_csv", required=True, help="Input SMILES CSV (with fragment_1|fragment_2)")
    parser.add_argument("--dist_file", required=True, help="Distance file containing min,max")
    parser.add_argument("--output_toml", default="sampling.toml", help="Output TOML config file")
    parser.add_argument("--slurm_script", default="submit_linkinvent.sh", help="SLURM script filename")
    parser.add_argument("--job_name", default="linkinvent", help="Job name for SBATCH")
    parser.add_argument("--mem", default="16G", help="Memory allocation (e.g., 16G)")
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per task")
    parser.add_argument("--gres", default="gpu:1", help="GPU resources (e.g., gpu:1)")
    parser.add_argument("--time", default=None, help="Walltime in D-HH:MM; if omitted and --pdb_path provided, will be estimated")
    parser.add_argument("--pdb_path", default=None, help="PDB file used to estimate walltime if --time not set")
    parser.add_argument("--account", default="def-aminpour", help="Compute Canada account string")
    parser.add_argument("--mail_user", default="jaharri1@ualberta.ca", help="Email for SLURM notifications")
    parser.add_argument("--mail_type", default="ALL", help="Notification type (e.g., ALL, END, FAIL)")
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.smiles_csv):
        raise FileNotFoundError(f"Missing SMILES file: {args.smiles_csv}")
    if not os.path.exists(args.dist_file):
        raise FileNotFoundError(f"Missing distance file: {args.dist_file}")

    generate_toml(args.smiles_csv, args.dist_file, args.output_toml)
    write_slurm_script(
        args.output_toml,
        slurm_script=args.slurm_script,
        job_name=args.job_name,
        output=f"{args.job_name}.out",
        error=f"{args.job_name}.err",
        gres=args.gres,
        mem=args.mem,
        cpus=args.cpus,
        time=args.time,
        account=args.account,
        mail_type=args.mail_type,
        mail_user=args.mail_user,
        pdb_path=args.pdb_path,
    )

if __name__ == "__main__":
    main()
