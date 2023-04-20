'''
Using DeepCore public oscillations data to reprocuce oscillations fit

Based on https://github.com/icecube/pisa/blob/master/pisa_examples/IceCube_3y_oscillations_example.ipynb
'''

from deimos.utils.plotting import *

from cptv_decoherence.analysis.pisa_analysis import IceCubeAnalysis

from pisa import ureg


#
# Main
#

if __name__ == "__main__" :


    #
    # Parse inputs
    #

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--real-data", action="store_true", help="Use real data (otherwise, unfluctuated pseudodata will be used)")
    parser.add_argument("-od", "--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("-np", "--num-points", type=int, required=False, default=50, help="Number of scan points (in each dimension)")
    args = parser.parse_args()


    #
    # Init analysis
    #

    analysis = IceCubeAnalysis(
        output_dir=args.output_dir,
        real_data=args.real_data,
    )    


    #
    # Scan oscillation parameters
    #

    # Define scan
    param_names = ["theta23", "deltam31"]
    # param_values = [[40., 45., 50.]*ureg["degree"], [2.4e-3, 2.5e-3, 2.6e-3]*ureg["eV**2"]]
    param_values = [np.linspace(35., 55., num=args.num_points)*ureg["degree"], np.linspace(2.2e-3, 2.7e-3, num=args.num_points)*ureg["eV**2"]]

    # Scan
    analysis.scan(param_names=param_names, param_values=param_values)

    #
    # Done
    #

    # Report
    print("\nOutput dir : %s\n" % analysis.output_dir)

