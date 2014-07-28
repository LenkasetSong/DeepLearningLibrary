namespace DeepLearningLibrary.Utility.Common
{
    using System;

    public abstract class Range<T>
    {
        private T min, max;

        public T Max
        {
            get { return max; }
            set { max = value; }
        }

        public T Min
        {
            get { return min; }
            set { min = value; }
        }

        public T Length;

        public Range(T min, T max)
        {
            this.min = min;
            this.max = max;
        }

        public abstract bool IsInside(T x);

        public bool IsInside(Range<T> range)
        {
            return ((IsInside(range.Min)) && (IsInside(range.Max)));
        }

        public bool IsOverlapping(Range<T> range)
        {
            return ((IsInside(range.min)) || (IsInside(range.max)));
        }
    }
}
