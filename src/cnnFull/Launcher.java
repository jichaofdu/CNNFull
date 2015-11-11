package cnnFull;
/**
 * �?��无误
 * @author Chao
 *
 */
public class Launcher {
	public static void main(String[] args){
//		NumberObject test = new NumberObject(1,28,28,"D:\\dataset\\");
		CNNClassification cnn = new CNNClassification();
		try {
			cnn.trainingProcedure();
			//cnn.testingProcedure();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
