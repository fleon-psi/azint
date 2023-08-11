import azint
import h5py
import sys
import numpy
import os

h5filename_in = sys.argv[1]
ponifilename = sys.argv[2]

n_bins_rad = 500
n_bins_azi = 16
pixel_split = 2
pixel_size = 75.0e-6

direct_beam_min_x = 1990
direct_beam_max_x = 2015
direct_beam_min_y = 2377
direct_beam_max_y = 2389

if len(sys.argv) >= 8:
    direct_beam_min_x = int(sys.argv[4])
    direct_beam_max_x = int(sys.argv[5])
    direct_beam_min_y = int(sys.argv[6])
    direct_beam_max_y = int(sys.argv[7])

direct_beam_ROI = (slice(direct_beam_min_y, direct_beam_max_y),
                   slice(direct_beam_min_x, direct_beam_max_x))

base_directory = "/sls/MX/Data10/e20757/"
output_directory = "/sls/MX/Data10/e20757/process/"

with h5py.File(base_directory + h5filename_in, 'r') as input_file:
    data_container = input_file["/entry/data/data"]
    nimages = data_container.shape[0]

    output_data = numpy.zeros((nimages, n_bins_azi, n_bins_rad))
    transmitted_beam = numpy.zeros(nimages)

    image_size = (data_container.shape[1], data_container.shape[2])

    if len(sys.argv) >= 4:
        with h5py.File(sys.argv[3], 'r') as file:
            bad_pix_map = numpy.array(file['data']).astype(bool)
    else:
        bad_pix_map = numpy.zeros(image_size)

    az = azint.AzimuthalIntegrator(ponifilename,
                                   shape=image_size,
                                   pixel_size=pixel_size,
                                   n_splitting=pixel_split,
                                   radial_bins=n_bins_rad,
                                   azimuth_bins=n_bins_azi,
                                   unit='q',
                                   mask=bad_pix_map,
                                   solid_angle=True,
                                   error_model=None,
                                   polarization_factor=None)

    az_no_corr = azint.AzimuthalIntegrator(ponifilename,
                                           shape=image_size,
                                           pixel_size=pixel_size,
                                           n_splitting=pixel_split,
                                           radial_bins=n_bins_rad,
                                           azimuth_bins=n_bins_azi,
                                           unit='q',
                                           mask=bad_pix_map,
                                           solid_angle=False,
                                           error_model=None,
                                           polarization_factor=None)

    for column_index in range(nimages):
        res = az.integrate(data_container[column_index, :, :])
        output_data[column_index, :, :] = res[0]
        if (direct_beam_min_x < direct_beam_max_x) and (direct_beam_min_y < direct_beam_max_y):
            transmitted_beam[column_index] = numpy.sum(data_container[(column_index, *direct_beam_ROI)])
        if column_index % 1000 == 0:
            print("Image: {}".format(column_index))

    output_dir_full = os.path.dirname(output_directory + h5filename_in)
    if not os.path.exists(output_dir_full):
        os.makedirs(output_dir_full)

    with h5py.File(output_directory + h5filename_in, 'w') as output_file:
        output_file.create_dataset('I_all', data=output_data)
        output_file.create_dataset('azi', data=az.azimuth_axis)
        output_file.create_dataset('q', data=az.radial_axis)
        if (direct_beam_min_x < direct_beam_max_x) and (direct_beam_min_y < direct_beam_max_y):
            output_file.create_dataset('I_t', data=transmitted_beam)
        output_file.create_dataset('n_pix', data=az_no_corr.norm.reshape(az_no_corr.output_shape))
        output_file.create_dataset('corr', data=az.corrections.reshape(image_size))
        output_file.create_dataset('mask', data=bad_pix_map)
