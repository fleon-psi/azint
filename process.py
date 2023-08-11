import azint
import h5py
import sys
import numpy

h5filename_in = sys.argv[1]
h5filename_out = sys.argv[2]
ponifilename = sys.argv[3]

n_bins_rad = 500
n_bins_azi = 16
pixel_split = 2

direct_beam_min_x = 1990
direct_beam_max_x = 2015
direct_beam_min_y = 2377
direct_beam_max_y = 2389

direct_beam_ROI = (slice(direct_beam_min_y, direct_beam_max_y),
                   slice(direct_beam_min_x, direct_beam_max_x))

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
                                   n_splitting=pixel_split,
                                   radial_bins=n_bins_rad,
                                   azimuth_bins=n_bins_azi,
                                   unit='q',
                                   mask=bad_pix_map,
                                   solid_angle=True,
                                   error_model=None,
                                   polarization_factor=None)

    for column_index in range(nimages):
        res = az.integrate(data_container[column_index, :, :])
        output_data[column_index, :, :] = res[0]
        transmitted_beam[column_index] = numpy.sum(data_container[(column_index, *direct_beam_ROI)])
        if column_index % 1000 == 0:
            print("Image: {}".format(column_index))

    with h5py.File(h5filename_out, 'w') as output_file:
        output_file.create_dataset('I_all', data=output_data)
        output_file.create_dataset('azi', data=az.azimuth_axis)
        output_file.create_dataset('q', data=az.radial_axis)
        output_file.create_dataset('I_t', data=transmitted_beam)
        output_file.create_dataset('n_pix', data=az.norm.reshape(az.output_shape))
        output_file.create_dataset('corr', data=az.corrections.reshape(az.output_shape))
        output_file.create_dataset('mask', data=bad_pix_map)
