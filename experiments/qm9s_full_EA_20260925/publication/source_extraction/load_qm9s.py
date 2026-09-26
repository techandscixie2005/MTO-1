"""NumPy-only iterator. Keeps raw labels and explicit masks; no automatic split."""
import pathlib
import numpy as np

def iter_molecules(root, require_normal=True):
    """Yield variable-size molecular records; require_normal=False includes failures."""
    for path in sorted((pathlib.Path(root)/'arrays').glob('part-*.npz')):
        with np.load(path,allow_pickle=False) as archive:
            # NpzFile does not cache decompressed members; load each array once.
            d={key:archive[key] for key in archive.files}
            for i in range(len(d['molecule_id'])):
                if require_normal and not (d['normal_termination'][i] and d['geometry_present'][i] and d['unambiguous_single_calculation'][i]):
                    continue
                am=d['atom_mask'][i]; sm=d['state_mask'][i]
                yield dict(molecule_id=int(d['molecule_id'][i]),
                    atomic_numbers=d['atomic_numbers'][i,am].copy(),
                    positions_angstrom=d['positions_angstrom'][i,am].copy(),
                    charge=float(d['charge'][i]),multiplicity=float(d['multiplicity'][i]),
                    state_index=d['state_index'][i,sm].copy(),
                    energy_eV=d['energy_eV'][i,sm].copy(),
                    wavelength_nm=d['wavelength_nm'][i,sm].copy(),
                    oscillator_strength=d['oscillator_strength'][i,sm].copy(),
                    transition_dipole_au=d['transition_dipole_au'][i,sm].copy(),
                    A_au2=d['A_au2'][i,sm].copy(),
                    scalar_label_mask=d['scalar_label_mask'][i,sm].copy(),
                    vector_label_mask=d['vector_label_mask'][i,sm].copy(),
                    normal_termination=bool(d['normal_termination'][i]),
                    response_convergence_reported=bool(d['response_convergence_reported'][i]))

def gaussian_spectrum(energy_eV, oscillator_strength, grid_eV, sigma_eV):
    """Optional area-normalized energy-domain density, with caller-chosen grid/width.

    No broadening was applied to the stored dataset. Result units: 1/eV.
    This is not an absolute absorption cross section or a wavelength density.
    """
    if sigma_eV<=0: raise ValueError('sigma_eV must be positive')
    e=np.asarray(energy_eV);f=np.asarray(oscillator_strength);g=np.asarray(grid_eV)
    valid=np.isfinite(e)&np.isfinite(f)
    return np.sum(f[valid,None]*np.exp(-0.5*((g[None,:]-e[valid,None])/sigma_eV)**2)/(np.sqrt(2*np.pi)*sigma_eV),axis=0)

if __name__=='__main__':
    import sys
    r=next(iter_molecules(sys.argv[1]))
    for k,v in r.items(): print(k, v.shape if isinstance(v,np.ndarray) else v)
