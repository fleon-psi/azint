import azint
import h5py
import sys
import numpy

h5filename_in = sys.argv[1]
h5filename_out = sys.argv[2]
ponifilename = sys.argv[3]

n_bins_rad = 500
n_bins_azi = 16

direct_beam_ROI = (slice(2377, 2389), slice(1990, 2015))

with h5py.File(h5filename_in, 'r') as input_file:
    data_container = input_file["/entry/data/data"]
    nimages = data_container.shape[0]

    output_data = numpy.zeros((nimages, n_bins_azi, n_bins_rad))
    transmitted_beam = numpy.zeros(nimages)

    image_size = (data_container.shape[1], data_container.shape[2])

    if len(sys.argv) == 5:
        with h5py.File(sys.argv[4], 'r') as file:
            bad_pix_map = numpy.array(file['data']).astype(bool)
    else:
        bad_pix_map = numpy.zeros(image_size)

    az = azint.AzimuthalIntegrator(ponifilename,
                                   shape=image_size,
                                   pixel_size=75.0e-06,
                                   n_splitting=2,
                                   radial_bins=n_bins_rad,
                                   azimuth_bins=n_bins_azi,
                                   unit='q',
                                   mask=bad_pix_map,
                                   solid_angle=True,
                                   polarization_factor=None)

    n_pix = az.integrate(bad_pix_map, None, False)[2]

    for column_index in range(nimages):
        res = az.integrate(data_container[column_index, :, :])
        output_data[column_index, :, :] = res[0]
        transmitted_beam[column_index] = numpy.sum(data_container[(column_index, *direct_beam_ROI)])

    with h5py.File(h5filename_out, 'w') as output_file:
        output_file.create_dataset('I_all', data=output_data)
        # output_file.create_dataset('phi_det', data=phi)
        # output_file.create_dataset('q', data=az.radia)
        output_file.create_dataset('i_t', data=transmitted_beam)
        output_file.create_dataset('n_pix', data=n_pix)
