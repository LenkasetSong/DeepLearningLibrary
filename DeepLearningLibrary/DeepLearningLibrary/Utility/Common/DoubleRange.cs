using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningLibrary.Utility.Common
{
    using System;

    public class DoubleRange : Range<double>
    {
        public DoubleRange(double min, double max) :
            base(min, max)
        { }

        public new Double Length
        {
            get { return Max - Min; }
        }

        public override bool IsInside(double x)
        {
            return ((x >= Min) && (x <= Max));
        }
    }
}
