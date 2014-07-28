﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningLibrary.Utility.ActivationFunction
{
    using System;

    public class SigmoidFunction : IActivationFunction
    {
        private double alpha = 2;

        public double Alpha
        {
            get { return alpha; }
            set { alpha = value; }
        }

        public SigmoidFunction() { }

        public SigmoidFunction(double alpha)
        {
            this.alpha = alpha;
        }

        public double Function(double x)
        {
            return (1 / (1 + Math.Exp(-alpha * x)));
        }

        public double Derivative(double x)
        {
            double y = Function(x);

            return (alpha * y * (1 - y));
        }

        public double Derivative2(double y)
        {
            return (alpha * y * (1 - y));
        }	
    }
}
