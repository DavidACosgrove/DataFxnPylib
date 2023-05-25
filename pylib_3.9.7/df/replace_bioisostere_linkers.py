#!/usr/bin/env python

# Use a bioisosteric linkers database to replace linkers in input
# molecules

import argparse
import concurrent.futures as cf
import random
import re
import sqlite3
import sys

from collections import defaultdict
from functools import reduce
from os import cpu_count
from pathlib import Path
from typing import Optional, Union

from rdkit import rdBase, Chem
from rdkit.Chem import rdmolops, rdDepictor

import df.find_linkers as fl
from df.int_range import IntRange

LINKER_COLORS = ['#14aadb', '#3beb24', '#000080', '#f0f060']


def parse_args(cli_args: list[str]):
    """
        Use argparse module to parse the command line arguments.

        Returns:
            (Namespace object): Parsed arguments in argparse Namespace
                                object.
        """
    parser = argparse.ArgumentParser(description='Replace linkers with'
                                                 ' bioisosteres.')
    parser.add_argument('-D', '--database-file', dest='db_file',
                        required=True,
                        help='Name of SQLite file containing the bioisosteric'
                             ' linkers.')
    in_args = parser.add_mutually_exclusive_group(required=True)
    in_args.add_argument('-S', '--query-smiles', dest='query_smiles',
                         help='Single SMILES string of molecule to be'
                              ' processed.  Will probably need to be in'
                              ' quotes.')
    in_args.add_argument('-I', '--input-file', dest='input_file',
                         help='Name of molecule file containing input'
                              ' molecules to be processed.')
    parser.add_argument('-O', '--output-file', dest='out_file',
                        required=True,
                        help='Name of output file.  Should be an SDF or'
                             ' SMILES file with extension .sdf or .smi.')
    parser.add_argument('--max-heavies', dest='max_heavies',
                        type=IntRange(1), default=8,
                        help='Maximum number of heavy atoms in linker,'
                             ' excluding the 2 ring atoms being linked.'
                             '  Default=%(default)s.')
    parser.add_argument('--max-bonds', dest='max_bonds',
                        type=IntRange(2), default=5,
                        help='Maximum number of bonds in shorted path between'
                             ' ring atoms being linked.  Default=%(default)s.')
    parser.add_argument('--plus-delta-length', dest='plus_length',
                        type=IntRange(-1, 5), default=-1,
                        help='Positive delta length for linker compared with'
                             ' that found in the query structure.  Default=-1'
                             ' means no maximum.')
    parser.add_argument('--minus-delta-length', dest='minus_length',
                        type=IntRange(-1, 5), default=-1,
                        help='Negative delta length for linker compared with'
                             ' that found in the query structure.  Default=-1'
                             ' means no minimum.')
    parser.add_argument('--match-donors', dest='match_donors',
                        action='store_true',
                        help='If True and the query linker has an hbond donor,'
                             ' the replacement must too, and not if not.  If'
                             ' False, it will take either.')
    parser.add_argument('--match-acceptors', dest='match_acceptors',
                        action='store_true',
                        help='If True and the query linker has an hbond'
                             ' acceptor, the replacement must too, and not if'
                             ' not.  If False, it will take either.')
    parser.add_argument('--no-ring-linkers', dest='no_ring_linkers',
                        action='store_true',
                        help='By default, small rings can be linkers.  This'
                             ' option removes that possibility.')
    parser.add_argument('--max-mols-per-input', dest='max_mols_per_input',
                        type=IntRange(-1), default=100,
                        help='Set a maximum number of products for each input'
                             ' molecule.  With several common linkers, the'
                             ' combinatorial explosion can make the results'
                             ' set too larger.  If the maximum is exceeded, a'
                             ' random selection of the requisite number is'
                             ' made.  Default=%(default)s, -1 means no'
                             ' maximum.')
    parser.add_argument('--max-total-output-mols', dest='max_total_mols',
                        type=IntRange(1), default=500000,
                        help='The maximum number of molecules that may be'
                             ' produced. Default=%(default)s.')
    parser.add_argument('--num-procs', dest='num_procs',
                        type=IntRange(1), default=cpu_count() - 1,
                        help='Number of processors to use for parallel'
                             ' processing of file.  Default=%(default)s.')

    args = parser.parse_args(cli_args)
    return args


def check_db_file(db_file: str):
    """
    Makes sure the db_file is a valid bioisostere one.
    Args:
        db_file:

    Returns:
        bool

    Raises:
         FileNotFoundError if db_file doesn't exist.
         ValueError if db_file doesn't contain the relevant tables.
    """
    # sqlite3 helpfully creates a db file if it doesn't exist!
    if not Path(db_file).exists():
        raise FileNotFoundError(f'{db_file} not available for reading.')

    conn = sqlite3.connect(db_file)
    for table in ['bioisosteres', 'linkers']:
        res = conn.execute('SELECT COUNT(name)'
                           ' FROM sqlite_schema'
                           ' WHERE type="table" AND name=?',
                           (table,)).fetchone()
        if res[0] == 0:
            raise ValueError(f'Invalid database {db_file}: no table "{table}".')


def make_new_smiles(mol_smi: str, linker_smi: str, bios: list[str]) -> list[str]:
    """
    Take the fragmented SMILES string, the SMILES of the current
    linker of interest and a list of bioisostere SMILES and make SMILES
    strings of molecules where the linker is replaced by the
    bioisosteres.  The new SMILES strings are not zipped up, because
    there may be more substitutions to make.
    Args:
        mol_smi: fragmented SMILES string
        linker_smi: SMILES string of current linker of interest
        bios: list of SMILES strings to be attached to left and right
              SMILES

    Returns:
        list of new SMILES strings, still in fragmented form.
    """
    params = rdmolops.MolzipParams()
    params.label = rdmolops.MolzipLabel.AtomMapNumber
    new_smis = []
    for b in bios:
        new_smi = mol_smi.replace(linker_smi, b)
        new_smis.append(new_smi)
        # print(f'"{new_smis[-1]}",')

    return new_smis


def alter_atom_maps(mol: Chem.Mol, new_maps: list[tuple[int, int]]):
    """
    Change the atom map numbers from the first in each tuple to the
    second. Assumes there's no overlap i.e. it won't be changing
    1 -> 2 and then 2 -> 3.
    """
    for a in mol.GetAtoms():
        for nm in new_maps:
            if a.GetAtomMapNum() == nm[0]:
                a.SetAtomMapNum(nm[1])
                break


def split_input_smiles(query_mol: Chem.Mol, linker_smis: list[str],
                       linker_atoms: list[list[int]],
                       max_heavies: int, max_bonds: int,
                       no_ring_linkers: bool,
                       linker_num: list[int]) -> Chem.Mol:
    """
    Take the molecule and split on any linkers to produce a
    fragmented SMILES with linkers and pieces, with the dummy map
    numbers adjusted so that multiple linkers don't have the same
    values.  Do it recursively so that the end product is one molecule
    split at all linkers.  fl.find_linkers returns a list of single
    linker breaks.
    Also, fill in the SMILES strings of the linkers in linker_smis,
    and the indices of the atoms from the initial molecule that are in
    those linkers.
    e.g. split c1ccccc1CCc1cnccc1OCOc1ccccc1 into
    [*:1]c1ccccc1.[*:1]CC[*:2].[*:2]c1cnccc1[*:3].[*:3]OCO[*:4].[*:4]c1ccccc1
    """
    new_mol = Chem.Mol(query_mol)
    _, linkers = fl.find_linkers((new_mol, ''), max_heavies=max_heavies,
                                 max_length=max_bonds,
                                 no_ring_linkers=no_ring_linkers)
    # print(f'depth = {len(linker_smis)} : linker num {linker_num[0]}')
    if not linkers:
        return query_mol
    # for l in linkers:
    #     print(l)

    linker_num[0] += 1
    new_maps = [(1, 2 * linker_num[0] - 1), (2, 2 * linker_num[0])]
    new_linker = Chem.Mol(linkers[0]._linker_mol)
    alter_atom_maps(new_linker, new_maps)
    new_left_mol = Chem.Mol(linkers[0]._left_mol)
    alter_atom_maps(new_left_mol, new_maps)
    new_right_mol = Chem.Mol(linkers[0]._right_mol)
    alter_atom_maps(new_right_mol, new_maps)

    linker_smis.append(Chem.MolToSmiles(new_linker))
    lats = []
    for a in new_linker.GetAtoms():
        try:
            lats.append(a.GetIntProp('InitIdx'))
        except KeyError:
            pass
    linker_atoms.append(lats)

    new_left_mol = split_input_smiles(new_left_mol, linker_smis, linker_atoms,
                                      max_heavies, max_bonds, no_ring_linkers,
                                      linker_num)
    new_right_mol = split_input_smiles(new_right_mol, linker_smis,
                                       linker_atoms, max_heavies, max_bonds,
                                       no_ring_linkers, linker_num)

    new_mol = Chem.RWMol()
    new_mol.InsertMol(new_left_mol)
    new_mol.InsertMol(new_linker)
    new_mol.InsertMol(new_right_mol)
    return new_mol


def get_dummy_map_nums_from_molecule(mol: Chem.Mol) -> list[int]:
    """
    Returns a sorted list of the map numbers of dummy atoms in the
    molecule.
    """
    dummies = []
    for a in mol.GetAtoms():
        if not a.GetAtomicNum():
            dummies.append(a.GetAtomMapNum())
    dummies.sort()
    return dummies


def add_linker_color_props(mol: Chem.Mol) -> None:
    """
    Add to the molecule the properties necessary for colouring the
    linkers in Spotfire.
    NB - not used at the moment, due to performance and memory issues.
    """
    linker_atoms = defaultdict(list)
    for a in mol.GetAtoms():
        try:
            linker_num = a.GetIntProp('Linker') - 1
            col_num = linker_num % len(LINKER_COLORS)
            linker_atoms[col_num].append(a.GetIdx())
            for nbr in a.GetNeighbors():
                linker_atoms[col_num].append(nbr.GetIdx())
        except KeyError:
            pass

    num_cols = max(linker_atoms.keys()) + 1
    col_recs = [list() for _ in range(2 * num_cols)]
    for col, atoms in linker_atoms.items():
        col_recs[col] = list(set(atoms))
        for a1 in col_recs[col]:
            for a2 in col_recs[col]:
                if a1 > a2:
                    bond = mol.GetBondBetweenAtoms(a1, a2)
                    if bond is not None:
                        col_recs[col + num_cols].append(bond.GetIdx())

    # The atom and bond numbers in Renderer_Highlight start from 1.
    high_str = ''
    for col in range(num_cols):
        at_list = ' '.join([str(a + 1) for a in col_recs[col]])
        bo_list = ' '.join([str(b + 1) for b in col_recs[col + num_cols]])
        high_str += f'COLOR {LINKER_COLORS[col]}\nATOMS {at_list}\nBONDS {bo_list}\n'

    if high_str:
        mol.SetProp('Renderer_Highlight', high_str)


def colour_input_mol(mol: Chem.Mol, linker_atoms: list[list[int]]) -> None:
    """
    linker_atoms contains the indices of the atoms that were identified
    as being in linkers.  Label them as such, and then add the
    appropriate colour properties.
    NB - not used at the moment, due to performance and memory issues.
    """
    for i, lats in enumerate(linker_atoms, 1):
        for lat in lats:
            atom = mol.GetAtomWithIdx(lat)
            atom.SetIntProp('Linker', i)

    add_linker_color_props(mol)


def fettled_linker_smis(linker_smi: str) -> list[str]:
    """
    Take the .-separated string of linker SMILES and return a list of
    the SMILES with the dummy atom numbers all changed to 1 and 2 as
    appropriate
    """
    out_smis = []
    for smi in linker_smi.split('.'):
        new1, new2 = extract_dummy_atoms_from_smiles(smi)
        out_smis.append(smi.replace(new1, '[*:1]').replace(new2, '[*:2]'))

    return out_smis


def zip_up_smiles(smis: list[str], linker_smis: list[str],
                  query_smi: str, num_to_have: int) -> tuple[list[str], list[list[str]]]:
    zipped_mols = []
    final_linker_smis = []
    zipped_smis = {}

    # Take a random selection if there are more molecules than
    # required.  Doing it this way allows for removal of duplicates.
    mols_to_have = [i for i in range(len(smis))]
    if num_to_have == -1 or len(smis) < num_to_have:
        num_to_have = len(smis)
    else:
        random.seed(10)
        random.shuffle(mols_to_have)

    for i in mols_to_have:
        new_mol = Chem.MolFromSmiles(smis[i])
        new_linker_smi = linker_smis[i]
        # print(new_linker_smis[i])

        zip_mol = rdmolops.molzip(new_mol)
        zip_smi = Chem.MolToSmiles(zip_mol)
        # Don't return the input molecule.
        if query_smi == zip_smi:
            continue
        # keep track of duplicates
        if zip_smi in zipped_smis:
            continue
        zipped_smis[zip_smi] = len(zipped_mols)
        zipped_mols.append(zip_smi)
        final_linker_smis.append(fettled_linker_smis(new_linker_smi))
        if len(zipped_mols) == num_to_have:
            break

    return zipped_mols, final_linker_smis


def collect_bioisosteres(linker_smis: list[str], db_file: Union[str, Path],
                         plus_length: int, minus_length: int,
                         match_donors: bool, match_acceptors: bool,
                         no_ring_linkers: bool) -> tuple[dict[str, list[str]], int]:
    """
    Pull together all the bioisosteres for the given linkers into a
    dict, and return it and the total number of analogues that would be
    produced.
    """
    repl_bios = dict()
    tot_mols = 1
    for lsmi in linker_smis:
        bios = fetch_bioisosteres(lsmi, db_file, plus_length, minus_length,
                                  match_donors, match_acceptors,
                                  no_ring_linkers)
        if lsmi not in bios:
            bios = [lsmi] + bios
        repl_bios[lsmi] = bios
        tot_mols *= len(bios)

    return repl_bios, tot_mols


def check_mol_can_be_used(mol: Chem.Mol) -> bool:
    """
    Make sure the mol is valid for linker replacement.
    For that it must be:
    Not None
    Have atoms.
    Not have any atom map indices, which will interfere
    with the zip later and won't produce a result.
    """
    if mol is None or not mol or not mol.GetNumAtoms():
        return False

    for atom in mol.GetAtoms():
        # GetAtomMapNum() returns 0 if the atom doesn't have a map number.
        if atom.GetAtomMapNum():
            return False

    return True


def estimate_output_size(query_mol: Chem.Mol, db_file: Union[str, Path],
                         max_heavies: int, max_bonds: int,
                         plus_length: int, minus_length: int,
                         match_donors: bool, match_acceptors: bool,
                         no_ring_linkers: bool) -> int:
    if not check_mol_can_be_used(query_mol):
        return 0

    query_cp = Chem.Mol(query_mol)
    for a in query_cp.GetAtoms():
        a.SetIntProp('InitIdx', a.GetIdx())

    linker_smis = []
    linker_atoms = []
    linker_num = [3]
    _ = split_input_smiles(query_cp, linker_smis, linker_atoms,
                           max_heavies, max_bonds, no_ring_linkers,
                           linker_num)

    _, tot_mols = \
        collect_bioisosteres(linker_smis, db_file, plus_length, minus_length,
                             match_donors, match_acceptors, no_ring_linkers)

    return tot_mols


def replace_linkers_via_smiles(query_mol: Chem.Mol, db_file: Union[str, Path],
                               max_heavies: int, max_bonds: int,
                               plus_length: int, minus_length: int,
                               match_donors: bool, match_acceptors: bool,
                               no_ring_linkers: bool,
                               max_mols_per_input: int,
                               max_total_mols: int) -> tuple[
    list[str], Union[None, str], Union[None, list[list[str]]]]:
    if not check_mol_can_be_used(query_mol):
        return [], None, None

    # Start by copying the molecule and labelling each atom with its
    # input index number so we can keep track of the atoms that go into
    # linkers without altering the input molecule.
    query_cp = Chem.Mol(query_mol)
    for a in query_cp.GetAtoms():
        a.SetIntProp('InitIdx', a.GetIdx())

    linker_smis = []
    linker_atoms = []
    linker_num = [3]
    split_mol = split_input_smiles(query_cp, linker_smis, linker_atoms,
                                   max_heavies, max_bonds, no_ring_linkers,
                                   linker_num)
    split_smi = Chem.MolToSmiles(split_mol)

    repl_bios, tot_mols = \
        collect_bioisosteres(linker_smis, db_file, plus_length, minus_length,
                             match_donors, match_acceptors, no_ring_linkers)

    # It just takes too long, sometimes, or it runs out of memory.
    if tot_mols > max_total_mols:
        print(f'Estimated number of molecules ({tot_mols}) too large.')
        return [], None, None

    new_smis = [split_smi]
    new_linker_smis = ['']

    for i, lsmi in enumerate(linker_smis, 1):
        next_new_smis = []
        next_new_linker_smis = []
        bios = repl_bios[lsmi]
        for new_smi, n_lnkr in zip(new_smis, new_linker_smis):
            for bio in bios:
                next_new_smis.append(new_smi.replace(lsmi, bio))
            if n_lnkr:
                next_new_linker_smis.extend([n_lnkr + '.' + b for b in bios])
            else:
                next_new_linker_smis.extend(bios)
        new_smis = next_new_smis
        new_linker_smis = next_new_linker_smis

    # for new_smi, n_lnkr in zip(new_smis, new_linker_smis):
    #     print(f'{new_smi} : {n_lnkr}')
    query_smi = Chem.MolToSmiles(query_mol)
    zipped_mols, final_linker_smis = zip_up_smiles(new_smis, new_linker_smis,
                                                   query_smi, max_mols_per_input)
    return zipped_mols, query_smi, final_linker_smis


def bulk_replace_linkers_via_smiles(mol_file: str, db_file: str,
                                    max_heavies: int, max_bonds: int,
                                    plus_length: int, minus_length: int,
                                    match_donors: bool, match_acceptors: bool,
                                    no_ring_linkers: bool, max_mols_per_input: int,
                                    max_total_mols, num_procs: int) -> tuple[
    Union[list[str], None], Union[list[str], None],
    Union[list[list[str]], None]]:
    """
    Take the structures in the mol file and process them with
    replace_linkers.  Returns None if file can't be read.
    New linkers must have a length (shortest distance in bonds between
    dummies) of l-minus_length to l+plus_length where l is the length
    of each linker in query_smiles.  If match_donors is True, then any
    replacement linkers must have a donor if the query linker does, and
    not if not, and likewise for the match_acceptors.  If either is
    False, it doesn't care.
    Returns a list of lists of all the new molecules and annotated
    copies of the originals.
    Args:
        mol_file:
        db_file:
        max_heavies:
        max_bonds:
        plus_length:
        minus_length:
        match_donors:
        match_acceptors:
        no_ring_linkers:
        max_mols_per_input:

    Returns:
        new molecules
    """
    suppl = fl.create_mol_supplier(mol_file)
    if suppl is None:
        return None, None, None

    all_new_mols = {}
    all_query_cps = {}
    all_linker_smis = {}
    parent_ids = []
    total_analogues = 0
    max_analogues = 0
    max_analogue_mol_name = None
    with cf.ProcessPoolExecutor(max_workers=num_procs) as pool:
        futures_to_mol_name = {}
        for i, mol in enumerate(suppl, 1):
            if not mol or not mol.GetNumAtoms():
                continue
            try:
                mol_name = mol.GetProp("_Name")
            except:
                mol_name = f'Str_{i}'
            parent_ids.append(mol_name)
            # Send it in and out of SMILES so it is in canonical SMILES order.
            mol_smi = Chem.MolToSmiles(mol)
            print(f'submitting {mol_smi}')
            mol = Chem.MolFromSmiles(mol_smi)
            mol.SetProp('_Name', mol_name)
            num_analogues = estimate_output_size(mol, db_file, max_heavies,
                                                 max_bonds, plus_length,
                                                 minus_length, match_donors,
                                                 match_acceptors,
                                                 no_ring_linkers)
            num_analogues = min(num_analogues, max_mols_per_input)
            if num_analogues > max_analogues:
                max_analogues = num_analogues
                max_analogue_mol_name = mol_name
            total_analogues += num_analogues
            if total_analogues > max_total_mols:
                print(f'Maximum number of analogues reached ({total_analogues}'
                      f' vs {max_total_mols}).')
                break

            fut = pool.submit(replace_linkers_via_smiles, mol, db_file, max_heavies,
                              max_bonds, plus_length, minus_length,
                              match_donors, match_acceptors, no_ring_linkers,
                              max_mols_per_input, max_total_mols)
            futures_to_mol_name[fut] = mol_name

        num_mols = 0
        for fut in cf.as_completed(futures_to_mol_name):
            mol_name = futures_to_mol_name[fut]
            new_mols, query_cp, linker_smis = fut.result()
            all_new_mols[mol_name] = new_mols
            all_query_cps[mol_name] = query_cp
            all_linker_smis[mol_name] = linker_smis
            num_mols += len(new_mols)

    print(f'Maximum number of analogues was {max_analogues} for'
          f' {max_analogue_mol_name}.')
    # put the output in input order
    new_new_mols = []
    new_parent_mols = []
    new_linker_smis = []
    for pid in parent_ids:
        try:
            if all_new_mols[pid]:
                new_new_mols.extend(all_new_mols[pid])
                new_parent_mols.extend([all_query_cps[pid]] * len(all_new_mols[pid]))
                new_linker_smis.extend(all_linker_smis[pid])
        except KeyError:
            # For some reason, such as there were no linkers in
            # the molecule, it produced no output.
            pass

    return new_new_mols, new_parent_mols, new_linker_smis


def trim_linkers_by_length(conn: sqlite3.Connection, query_linker: str,
                           linkers: list[str], plus_length: int,
                           minus_length) -> list[str]:
    """
    Remove any linkers that have length outside the deltas given.
    """
    sql1 = """SELECT DISTINCT path_length, linker_smiles
     FROM linkers WHERE linker_smiles = ?"""

    row = conn.execute(sql1, (query_linker,)).fetchone()
    if row is None:
        return []

    if plus_length == -1:
        max_length = 1000
    else:
        max_length = row[0] + plus_length
    if minus_length == -1:
        min_length = 0
    else:
        min_length = row[0] - minus_length

    sql2 = f"""SELECT DISTINCT path_length, linker_smiles
     FROM linkers WHERE
      linker_smiles IN ({','.join(['?' for _ in range(len(linkers))])})"""
    new_linkers = []
    for row in conn.execute(sql2, linkers):
        if min_length <= row[0] <= max_length:
            new_linkers.append(row[1])
    return new_linkers


def trim_linkers_by_hbonding(conn: sqlite3.Connection, query_linker: str,
                             linkers: list[str], match_donors: bool,
                             match_acceptors: bool) -> list[str]:
    """
    Remove any linkers that fail the hbonding tests.
    """
    sql1 = """SELECT DISTINCT num_donors, num_acceptors, linker_smiles
     FROM linkers WHERE linker_smiles = ?"""

    row = conn.execute(sql1, (query_linker,)).fetchone()
    if row is None:
        return []
    num_donors = row[0]
    num_acceptors = row[1]

    sql2 = f"""SELECT DISTINCT num_donors, num_acceptors, linker_smiles
     FROM linkers WHERE
      linker_smiles IN ({','.join(['?' for _ in range(len(linkers))])})"""
    new_linkers = []
    for row in conn.execute(sql2, linkers):
        if match_donors:
            if not ((num_donors and row[0])
                    or (not num_donors and not row[0])):
                continue
        if match_acceptors:
            if not ((num_acceptors and row[1])
                    or (not num_acceptors and not row[1])):
                continue
        new_linkers.append(row[2])
    return new_linkers


def extract_dummy_atoms_from_smiles(smiles: str) -> tuple[str, str]:
    """
    Extract the [*:X] strings from the linker SMILES.
    """
    dummy_regex = re.compile(r'\[\*:(\d+)\]')
    dummy_matches = sorted([int(dm) for dm in dummy_regex.findall(smiles)])
    dummy1 = f'[*:{dummy_matches[0]}]'
    dummy2 = f'[*:{dummy_matches[1]}]'
    return dummy1, dummy2


def fetch_bioisosteres(linker_smi: str, db_file: str,
                       plus_length: int, minus_length: int,
                       match_donors: bool, match_acceptors: bool,
                       no_ring_linkers: bool) -> Optional[list[str]]:
    """
    Pull all bioisosteres from db_file that include the linker_smi
    on one side or the other.  Returns SMILES strings of the
    swaps.  Assumes, but doesn't check, that linker_smi doesn't contain
    [*:1] or [*:2].  The results will most likely be wrong if it does.
    In normal use by this script, the lowest the atom numbers should be
    is 3 and 4.
    Bioisosteres must have a length (shortest distance in bonds between
    dummies) of l-minus_length to l+plus_length where l is the length
    of each linker in query_smiles.  If match_donors is True, then any
    replacement linkers must have a donor if the query linker does, and
    not if not, and likewise for the match_acceptors.  If either is
    False, it doesn't care.
    Args:
        linker_smi: assumed to contain 2 dummies for the link points
                    e.g. [*:1]C[*:2] and be in canonical SMILES to
                    match the linkers in the db_file.
        db_file: name of valid SQLite3 database.
        plus_length:
        minus_length:
        match_donors:
        match_acceptors:
        no_ring_linkers:

    Returns:
        SMILES strings in a list

    Raises: FileNotFoundError if db_file doesn't exist.
    """
    check_db_file(db_file)
    # print(f'fetch_bioisosteres : {linker_smi}')
    # the linker_smi won't necessarily have the dummies with map
    # numbers 1 and 2, but they need to be so for looking up.
    new1, new2 = extract_dummy_atoms_from_smiles(linker_smi)
    # Left and right are completely arbitrary for this - a minor change
    # to a molecule can change the canonical order of the atoms such
    # that the linker comes out with the reversed atom maps. Thus,
    # search for both ways round.
    mended_smi1 = linker_smi.replace(new1, '[*:1]').replace(new2, '[*:2]')
    mended_smi2 = linker_smi.replace(new1, '[*:2]').replace(new2, '[*:1]')

    conn = sqlite3.connect(db_file)
    sql = """SELECT linker1_smiles, linker2_smiles FROM bioisosteres
    WHERE linker1_smiles = ? OR linker2_smiles = ?
    OR linker1_smiles = ? OR linker2_smiles = ?"""
    linkers = conn.execute(sql, (mended_smi1, mended_smi1, mended_smi2,
                                 mended_smi2)).fetchall()
    if not linkers:
        return []

    bios = []
    for linker in linkers:
        if linker[0] == mended_smi1 or linker[0] == mended_smi2:
            bios.append(linker[1])
        else:
            bios.append(linker[0])

    # For the trimming, only one of the fragments need be examined, as
    # the properties being tested are invariant on reversal.
    bios = trim_linkers_by_length(conn, mended_smi1, bios, plus_length,
                                  minus_length)

    if match_donors or match_donors:
        bios = trim_linkers_by_hbonding(conn, mended_smi1, bios, match_donors,
                                        match_acceptors)

    # Now replace the :1 and :2 dummies with the ones we require for this
    # molecule. In the case of an asymmetric linker, we have no information
    # about whether this should be one way or the other, so use them both.
    final_bios = []
    for b in bios:
        final_bios.append(b.replace('[*:1]', new1).replace('[*:2]', new2))
        final_bios.append(b.replace('[*:2]', new1).replace('[*:1]', new2))

    # symmetrical ones will have given the same thing twice, so remove them,
    # and enforce no_ring_linkers at the same time since both require a
    # molecule object to be created.
    new_final_bios = []
    for b in final_bios:
        bmol = Chem.MolFromSmiles(b)
        if no_ring_linkers:
            ring = False
            for atom in bmol.GetAtoms():
                if atom.IsInRing():
                    ring = True
                    break
            if ring:
                continue
        new_final_bios.append(Chem.MolToSmiles(bmol))

    # It's not essential that final_bios is sorted, but it makes testing
    # easier if the order is always consistent.
    final_bios = sorted(list(set(new_final_bios)))
    return final_bios


def write_mols(mol_lists: list[str], out_file: str) -> bool:
    """
    Writes the molecules to the named file.  Returns bool on success.
    Args:
        mol_lists:
        out_file:

    Returns:

    """
    out_path = Path(out_file)
    if out_path.suffix == '.smi':
        writer = Chem.SmilesWriter(out_file, includeHeader=False)
    elif out_path.suffix == '.sdf':
        writer = Chem.SDWriter(out_file)
    else:
        print(f'ERROR : unrecognised extension for file {out_file}. Nothing'
              f' written.')
        return False

    for smi in mol_lists:
        if smi is not None:
            mol = Chem.MolFromSmiles(smi)
            writer.write(mol)

    return True


def main(cli_args):
    print(f'Using RDKit version {rdBase.rdkitVersion}.')
    # so that properties, such as the _Name, are pickled when passed
    # into the multiprocessing bit.  Passing properties back out
    # requires this to be done in the sub-process as well.
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    args = parse_args(cli_args)
    if args is None:
        return False

    if args.query_smiles is not None:
        query_mol = Chem.MolFromSmiles(args.query_smiles)
        if query_mol is None:
            print(f'Bad input SMILES {args.query_smiles}')
            return False
        new_mols, query_cp, _ = \
            replace_linkers_via_smiles(query_mol, args.db_file, args.max_heavies,
                                       args.max_bonds, args.plus_length,
                                       args.minus_length, args.match_donors,
                                       args.match_acceptors, args.no_ring_linkers,
                                       args.max_mols_per_input, args.max_total_mols)
    else:
        new_mols, query_cps, _ = \
            bulk_replace_linkers_via_smiles(args.input_file, args.db_file,
                                            args.max_heavies, args.max_bonds,
                                            args.plus_length, args.minus_length,
                                            args.match_donors, args.match_acceptors,
                                            args.no_ring_linkers, args.max_mols_per_input,
                                            args.max_total_mols, args.num_procs)

    if new_mols is None or not new_mols:
        return False

    if not write_mols(new_mols, args.out_file):
        return False


if __name__ == '__main__':
    sys.exit(not main(sys.argv[1:]))
