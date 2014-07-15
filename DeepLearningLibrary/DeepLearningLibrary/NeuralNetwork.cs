using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace DeepLearningLibrary
{
    public delegate double ActivationFunc(double input);

    public class NeuralNetwork
    {
        private List<NeuralLayer> layers;
        internal List<NeuralLayer> Layers
        {
            get { return layers; }
            set { layers = value; }
        }

        private ActivationFunc func;
        public ActivationFunc Func
        {
            get { return func; }
            set { func = value; }
        }

        public NeuralNetwork(int[] num_layers, ActivationFunc _func)
        {
            func = _func;

            initLayers(num_layers);
        }

        private void initLayers(int[] num_layers)
        {
            layers = new List<NeuralLayer>();

            for (int i = 0; i < num_layers.Length - 1; i++)
            {
                Matrix<double> weight = new SparseMatrix(num_layers[i], num_layers[i + 1]);

                layers.Add(new NeuralLayer(weight, func));
            }
        }

        public Matrix<double> forwardFeed(Matrix<double> input)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                input = layers[i].forwardFeed(input);
            }

            return layers[layers.Count - 1].Output;
        }

        public void backPropagate(Matrix<double> output)
        {
            Matrix<double> sig = output.Add(layers[layers.Count - 1].Output.Multiply(-1));

            for (int i = layers.Count - 1; i >= 0; i--)
            {
                sig = layers[i].backPropagate(sig);
            }
        }

        public void updateDerivative()
        {
            foreach (var layer in layers)
                layer.updateDerivative();
        }

        public void train(Matrix<double> input, Matrix<double> output)
        {
            forwardFeed(input);
            backPropagate(output);
            updateDerivative();
        }

        public void applyDerivative(int m, double miu, double alpha)
        {
            foreach (var layer in layers)
                layer.applyDerivative(m, miu, alpha);
        }


    }
}