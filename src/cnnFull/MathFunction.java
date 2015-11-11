package cnnFull;

public class MathFunction {
	
	public static double sigmoid(double input){
		return 1.0 / (1.0 + Math.exp(0 - input));
	}

	public static double[][] convolutional(Maps map, ConvolutionalKernel ck){	
		int width = map.getMatrix().length;
		int newWidth = (width - 5) + 1;
		double[][] resultMatrix = new double[newWidth][newWidth];
		for(int i = 0; i < newWidth; i++){
			for(int j = 0; j < newWidth; j++){
				double temp = 0;
				for(int m = 0; m < 5; m++){
					for(int n = 0; n < 5; n++){
						temp += map.getMatrix()[i + m][j + n] * ck.getWeight(m, n);
					}
				}
				resultMatrix[i][j] = MathFunction.sigmoid(temp);
			}
		}
		return resultMatrix;
	}
	

	public static double[][] subSampling(Maps map){
		int width = map.getMatrix().length;
		int newWidth = width / 2;
		double[][] resultMatrix = new double[newWidth][newWidth];
		for(int i = 0 ; i < newWidth; i++){
			for(int j = 0; j < newWidth; j++){
				double temp = map.getMatrix()[i * 2 ][j * 2 ] +
						      map.getMatrix()[i * 2 + 1][j * 2 ] + 
						      map.getMatrix()[i * 2 ][j * 2 + 1] +
						      map.getMatrix()[i * 2 + 1][j * 2 + 1];
				resultMatrix[i][j] = temp / 4;
			}
		}
		return resultMatrix;
	}
	

	public static double[][] calculateConv2(Maps map, ConvolutionalKernel ck){
		int width = map.getMatrix().length;
		int newWidth = (width - 5) + 1;
		double[][] resultMatrix = new double[newWidth][newWidth];
		for(int i = 0; i < newWidth; i++){
			for(int j = 0; j < newWidth; j++){
				double temp = 0;
				for(int m = 0; m < 5; m++){
					for(int n = 0; n < 5; n++){
						temp += map.getMatrix()[i + m][j + n] * ck.getWeight(m, n);
					}
				}
				resultMatrix[i][j] = temp;
			}
		}
		return resultMatrix;
	}
	

	public static double[][] upSampling(double[][] error){
		int width = error.length;
		int newWidth = width * 2;
		double[][] resultMatrix = new double[newWidth][newWidth];
		for(int i = 0; i < width; i++){
			for(int j = 0; j < width; j++){
				double temp = error[i][j] / 4;
				resultMatrix[i * 2][j * 2] = temp;
				resultMatrix[i * 2 + 1][j * 2] = temp;
				resultMatrix[i * 2][j * 2 + 1] = temp;
				resultMatrix[i * 2 + 1][j * 2 + 1] = temp;
			}
		}
		return resultMatrix;
	}
	
}
