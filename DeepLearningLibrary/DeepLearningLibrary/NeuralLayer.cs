using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace DeepLearningLibrary
{
    class NeuralLayer
    {
        private Matrix<double> weight;

        public Matrix<double> Weight
        {
            get { return weight; }
            set { weight = value; }
        }

        private Matrix<double> output;

        public Matrix<double> Output
        {
            get { return output; }
            set { output = value; }
        }

        private Matrix<double> sig;
        public Matrix<double> Sig
        {
            get { return sig; }
            set { sig = value; }
        }

        private Matrix<double> der;
        public Matrix<double> Der
        {
            get { return der; }
            set { der = value; }
        }

        private Matrix<double> der_weight;

        public Matrix<double> Der_weight
        {
            get { return der_weight; }
            set { der_weight = value; }
        }

        private ActivationFunc func;

        public ActivationFunc Func
        {
            get { return func; }
            set { func = value; }
        }

        private Matrix<double> input;

        public Matrix<double> Input
        {
            get { return input; }
            set { input = value; }
        }

        public NeuralLayer(double[][] _weight, ActivationFunc _func)
        {
            setWeight(_weight);
            func = _func;
            der_weight = new SparseMatrix(weight.RowCount, weight.ColumnCount);
        }

        public NeuralLayer(Matrix<double> _weight, ActivationFunc _func)
        {
            weight = _weight;
            func = _func;
            der_weight = new SparseMatrix(weight.RowCount, weight.ColumnCount);
        }

        public void setWeight(double[][] _weight)
        {
            weight = new SparseMatrix(_weight.Length, _weight[0].Length);
            for (int i = 0; i < _weight.Length; i++)
            {
                weight.SetRow(i, _weight[i]);
            }
        }

        public Matrix<double> forwardFeed(Matrix<double> _input)
        {
            input = _input;

            Func<double, double> f = new Func<double, double>(func);

            output = input.Multiply(weight).Map(f);

            return output;
        }

        public Matrix<double> backPropagate(Matrix<double> sig_k)
        {
            sig = input.PointwiseMultiply(sig_k.Multiply(weight.Transpose()));

            der = input.Transpose().Multiply(sig_k);

            return sig;
        }

        public void updateDerivative()
        {
            der_weight = der_weight.Add(der);
        }

        public void applyDerivative(int m, double miu, double alpha)
        {
            Matrix<double> tmp = der.Multiply(1 / m).Add(weight.Multiply(miu));

            weight = weight.Add(tmp.Multiply(-alpha));
        }
    }
}
