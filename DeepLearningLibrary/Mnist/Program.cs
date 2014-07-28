using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using DeepLearningLibrary.Networks;
using DeepLearningLibrary.Utility.ActivationFunction;
using DeepLearningLibrary.Utility.Learning;

namespace Mnist
{
    class Program
    {
        static void Main()
        {
            double[][] inputs = new double[8][];
            double[][] outputs = new double[8][];

            for (int i = 0; i < 8; i++)
            {
                inputs[i] = new double[2];
                outputs[i] = new double[1];
            }

            inputs[0][0] = 0;
            inputs[0][1] = 1;

            outputs[0][0] = 0;

            inputs[1][0] = 1;
            inputs[1][1] = 1;

            outputs[1][0] = 0;

            inputs[2][0] = 3;
            inputs[2][1] = 4;

            outputs[2][0] = 0;

            inputs[3][0] = 1;
            inputs[3][1] = 7;

            outputs[3][0] = 1;

            inputs[4][0] = 2;
            inputs[4][1] = 11;

            outputs[4][0] = 1;

            inputs[5][0] = 4;
            inputs[5][1] = 10;

            outputs[5][0] = 1;

            inputs[6][0] = 4;
            inputs[6][1] = 5;

            outputs[6][0] = 0;

            inputs[7][0] = 5;
            inputs[7][1] = 12;

            outputs[7][0] = 1;

            SigmoidFunction function = new SigmoidFunction(0.01);

            ActivationNetwork network = new ActivationNetwork(function, 2, 20, 1);

            BackPropagationLearning bp = new BackPropagationLearning(network);

            bp.Momentum = 0.5;
            bp.LearningRate = 1;

            double error = 1;
            double[] test = {1,3.5};

            while(error > 0.05)
            {
                error = bp.RunEpoch(inputs, outputs);

                Console.WriteLine(error);
            }

            double prob = network.Compute(test)[0];

            Console.WriteLine(prob);

            if(prob > 0.5)
            {
                Console.WriteLine(1);
            }
            else
            {
                Console.WriteLine(0);
            }
        } 
    } 
}
