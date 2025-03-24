package JFlow.data;

class DataUtility {
    
    public static double max(double[][][] array) {
        double max = Double.NEGATIVE_INFINITY;

        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        for (int i = 0; i < channels; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    max = Math.max(max, array[i][j][k]);
                }
            }
        }

        return max;
    }

    public static double[][][] multiply(double[][][] array, double scalar) {
        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        double[][][] result = new double[channels][height][width];

        for (int i = 0; i < channels; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result[i][j][k] =  array[i][j][k] * scalar;
                }
            }
        }
        return result;
    }

    public static double[][][] add(double[][][] array, double scalar) {
        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        double[][][] result = new double[channels][height][width];

        for (int i = 0; i < channels; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result[i][j][k] =  array[i][j][k] + scalar;
                }
            }
        }
        return result;
    }
}
