using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using DeepLearningLibrary.Networks;
using DeepLearningLibrary.Layers;
using DeepLearningLibrary.Utility.ActivationFunction;
using DeepLearningLibrary.Utility.Learning;

namespace Mnist
{
    class Program
    {
        static void Main()
        {
            double[][] inputs = new double[4][];

            for (int i = 0; i < 4; i++)
            {
                inputs[i] = new double[2];
            }

            inputs[0][0] = 1;
            inputs[0][1] = 0;

            inputs[1][0] = 0;
            inputs[1][1] = 1;

            SigmoidFunction function = new SigmoidFunction(0.01);

            RBMLayer rbm = new RBMLayer(3, 3, function);

            RBMLearning bp = new RBMLearning(rbm);

            bp.K = 10;

            bp.Momentum = 0.5;
            bp.LearningRate = 1;

            double[][] test = inputs;

            for(int i = 0; i < 100000; i++)
            {
                bp.RunEpoch(inputs);
            }

            for (int i = 0; i < test.Length; i++)
            {
                double[] output = rbm.ComputeDown(inputs[i]);

                Console.WriteLine("{0}, {1}, {2}", output[0], output[1], output[2]);
            }
        } 
    } 
}
