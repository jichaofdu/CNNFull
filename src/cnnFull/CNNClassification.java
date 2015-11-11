package cnnFull;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Random;
//主程序
public class CNNClassification {
	private Maps[] inputLayer;
	private Maps[] c1Layer;
	private Maps[] s2Layer;
	private Maps[] c3Layer;
	private Maps[] s4Layer;
	private Maps[] c5Layer;
	private double[] gapLine;
	private double[] gapError;
	private Maps[] f6Layer;
	private Maps[] outputLayer;
	
	private ConvolutionalKernel[] ck1;
	private SubsampleKernel[] sk2;//没有用到
	private ConvolutionalKernel[][] ck3Set;
	private SubsampleKernel[] sk4;//没有用到
	private BPParameter[] gap5IH;
	private BPParameter[] bp6HH;
	private BPParameter[] bp7HO;
	
	private int learningTimes;
	private int trainingSetSize;
	private int testSetSize;
	private int ckSize;
	private double learningRate;
	
	private int desiredNumber;
	private double[] desiredOutput;
	private int guessNumber;
	
	private String weightSavePath = "d:\\PJ1\\part2\\data\\";
	private String dataPath = "D:\\newTestSet\\";
	private String resultPath = "d:\\Test_Set_Result.txt";
	private String logPath = "d:\\Log_Record.txt";
	
	public CNNClassification(){
		this.learningTimes = HyperParameter.LEARNINGTIMES;
		this.trainingSetSize = HyperParameter.TRAININGSETSIZE;
		this.testSetSize = HyperParameter.TESTSETSIZE;
		this.ckSize = HyperParameter.CKSIZE;
		this.learningRate = HyperParameter.learingRate;
		initPara();
		generateLayers();
	}	
	
	public void trainingProcedure() throws IOException{
		LogRecord.logRecord("[Tip] Initialize Layers Success.",logPath);
		initPara();
		LogRecord.logRecord("[Tip] Initialize Weight Success.",logPath);
		for(int times = 0;times < learningTimes;times++){
			//For every times of training.
			for(int i = 1;i <= trainingSetSize;i++){
				NumberManager numberObj = new NumberManager(i,28,28,dataPath);
				setNowCase(numberObj);
				calculateOutput();
				guessNumberAndSaveAnswer();
				backPropagation();
				reset();
				if((double)(i % 100) == 0 && i >= 100){
					System.out.println("["+ i + "] Runing......");
				}
				if((double)(i % 1000) == 0 && i >= 1000){
					saveParaToDisk();
					LogRecord.logRecord("[" + i + "] Saved weight to the disk. ",logPath);
				}
			}
		}
		LogRecord.logRecord("[End] Learning Procedure End ",logPath);
		int correct = 0;
		for(int i = 1;i < 1 + testSetSize;i++){
			NumberManager numberObj = new NumberManager(i,28,28,dataPath);
			setNowCase(numberObj);
			calculateOutput();
			guessNumberAndSaveAnswer();
			if(desiredNumber == guessNumber){
				correct++;
			}
		}
		System.out.println("[End] Correct rate:" + correct + " / " + testSetSize);
		LogRecord.logRecord("[End] Correct rate:" + correct + " / " + testSetSize,logPath);
	}
	
	public void testingProcedure() throws FileNotFoundException, ClassNotFoundException, IOException{
		readParaFromDisk();
		int correct = 0;
		for(int i = 1;i < 1 + testSetSize;i++){
			NumberManager numberObj = new NumberManager(i,28,28,"D:\\newTestSet\\");
			setNowCase(numberObj);
			calculateOutput();
			guessNumberAndSaveAnswer();
			if(desiredNumber == guessNumber){
				correct++;
			}
		}
		System.out.println("[End] Correct rate:" + correct + " / " + testSetSize);
		LogRecord.logRecord("[End] Correct rate:" + correct + " / " + testSetSize,logPath);
	}
	
	//To init all the para
	private void initPara(){
		Random randomgen = new Random();
		desiredOutput = new double[10];
		//Initialize weight and bias between intput and C1
		double[][] ck1TempWeight = new double[ckSize][ckSize];
		double ck1TempBias;
		ck1 = new ConvolutionalKernel[6];
		for(int i = 0;i < ck1.length;i++){
			for(int j = 0;j < ckSize;j++){
				for(int k = 0;k < ckSize;k++){
					ck1TempWeight[j][k] = (randomgen.nextDouble() - 0.5) * 2;
				}
			}
			ck1TempBias = (randomgen.nextDouble() - 0.5) * 2 * 0.2;
			ck1[i] = new ConvolutionalKernel(ckSize,ck1TempWeight,ck1TempBias,1,"ck1"+i);
		}
		//Initialize weight and bias between C1 and S2
			//初始化下采样参数，这部分没用到
		//Initialize weight and bias between S2 and C3
		double[][] ck3TempWeight = new double[ckSize][ckSize];
		double ck3TempBias;
		ck3Set = new ConvolutionalKernel[16][6];
		for(int i = 0;i < ck3Set.length;i++){
			for(int lineIndex = 0;lineIndex < ck3Set[i].length;lineIndex++){
				for(int j = 0;j < ckSize;j++){
					for(int k = 0;k < ckSize;k++){
						ck3TempWeight[j][k] = (randomgen.nextDouble() - 0.5) * 2;
					}
				}
				ck3TempBias = (randomgen.nextDouble() - 0.5) * 2 * 0.2;
				ck3Set[i][lineIndex] = new ConvolutionalKernel(ckSize,ck3TempWeight,ck3TempBias,1,"ck3"+i+""+lineIndex);
			}

		}
		//Initialize weight and bias between C3 and S4
			//初始化下采样参数，这部分没用到
		//Initialize weight and bias between S4 and C5
		int gapLineLength = 16 * 5 * 5;
		double[] gapTempWeight = new double[gapLineLength];
		double gapTempBias;
		gap5IH = new BPParameter[120];
		for(int i = 0;i < gap5IH.length;i++){
			for(int j = 0;j < gapLineLength;j++){
				gapTempWeight[j] = ((randomgen.nextDouble() - 0.5) * 2) ;
			}
			gapTempBias = randomgen.nextDouble() - 1.0d;
			gap5IH[i] = new BPParameter(gapLineLength,gapTempWeight,gapTempBias,i,"IH");
		}
		//Initialize weight and bias between C5 and F6
		double[] bp1TempWeight = new double[gap5IH.length];
		double bp1TempBias;
		bp6HH = new BPParameter[84];
		for(int i = 0;i < bp6HH.length;i++){
			for(int j = 0;j < gap5IH.length;j++){
				bp1TempWeight[j] = (randomgen.nextDouble() - 0.5) * 2 ;
			}
			bp1TempBias = (randomgen.nextDouble() - 0.5) * 2 ;
			bp6HH[i] = new BPParameter(gap5IH.length,bp1TempWeight,bp1TempBias,i,"HH");
		}
		//Initialize weight and bias between F6 and output
		double[] bp2TempWeight = new double[bp6HH.length];
		double bp2TempBias;
		this.bp7HO = new BPParameter[10];
		for(int i = 0;i < bp7HO.length;i++){
			for(int j = 0;j < bp6HH.length;j++){
				bp2TempWeight[j] = (randomgen.nextDouble() - 0.5) * 2 ;
			}
			bp2TempBias = (randomgen.nextDouble() - 0.5) * 2;
			bp7HO[i] = new BPParameter(bp6HH.length,bp2TempWeight,bp2TempBias,i,"HO");
 		}
	}
	
	//To generate all the layers and give the init number.
	private void generateLayers(){
		inputLayer = new Maps[1];
		for(int i = 0;i < 1;i++){
			inputLayer[i] = new Maps(32);
		}
		c1Layer = new Maps[6];
		for(int i = 0;i < c1Layer.length;i++){
			c1Layer[i] = new Maps(28);
		}
		s2Layer = new Maps[6];
		for(int i = 0;i < s2Layer.length;i++){
			s2Layer[i] = new Maps(14);
		}
		c3Layer = new Maps[16];
		for(int i = 0;i < c3Layer.length;i++){
			c3Layer[i] = new Maps(10);
		}
		s4Layer = new Maps[16];
		for(int i = 0;i < s4Layer.length;i++){
			s4Layer[i] = new Maps(5);
		}
		c5Layer = new Maps[120];
		for(int i = 0;i < c5Layer.length;i++){
			c5Layer[i] = new Maps(1);
		}
		f6Layer = new Maps[84];
		for(int i = 0;i < f6Layer.length;i++){
			f6Layer[i] = new Maps(1);
		}
		outputLayer= new Maps[10];
		for(int i = 0;i < outputLayer.length;i++){
			outputLayer[i] = new Maps(1);
		}
		gapError = new double[400];
		gapLine = new double[400];
	}
	
	//To set the case that will calculate.
	private void setNowCase(NumberManager nb){
		for(int i = 0;i < 32;i++){
			for(int j = 0;j < 32;j++){
				inputLayer[0].setNumber(i, j, nb.getValue(i, j));
			}
		}
		desiredNumber = nb.getActualNumber();
		for(int i = 0;i < 10;i++){
			desiredOutput[i] = 0;
		}
		if(desiredNumber >= 0){
			desiredOutput[desiredNumber] = 1;
		}
	}
	
	private void calculateOutput(){
		//Calculate NO.1 layer: c1 layer
		for(int i = 0; i <c1Layer.length; i++){
			c1Layer[i].setMatrix(MathFunction.convolutional(inputLayer[0], ck1[i]));
		}
		//Calcualte NO.2 layer: s2 layer
		for(int i = 0; i < s2Layer.length; i++){
			s2Layer[i].setMatrix(MathFunction.subSampling(c1Layer[i]));
		}
		//Calculate NO.3 layer: c3 layer
		double[][][][] temp = new double[6][16][10][10];
		double[][][] tempMap2 = new double[16][10][10];
		for(int i = 0; i < s2Layer.length; i++){
			for(int j = 0; j < c3Layer.length; j++){
				temp[i][j] = MathFunction.calculateConv2(s2Layer[i], ck3Set[j][i]);
			}
		}
		for(int j = 0; j < c3Layer.length; j++){
			for(int m = 0; m < c3Layer[j].getMatrix().length; m++){
				for(int n = 0; n < c3Layer[j].getMatrix().length; n++){
					double number = 0;
					for(int i = 0; i < s2Layer.length; i++){
						number += temp[i][j][m][n];
					}
					tempMap2[j][m][n] = MathFunction.sigmoid(number);
				}
			}
		}
		for(int i = 0; i < c3Layer.length; i++){
			c3Layer[i].setMatrix(tempMap2[i]);
		}
		//Calculate NO.4 layer: s4 layer
		for(int i = 0; i < s4Layer.length; i++){
			s4Layer[i].setMatrix(MathFunction.subSampling(c3Layer[i]));
		}
		//Calculate NO.5 layer: c5 layer  
		for(int i = 0; i < 16; i++){
			for(int m = 0; m < 5; m++){
				for(int n = 0; n < 5; n++){
					gapLine[i * 25 + m * 5 + n] = s4Layer[i].getNumber(m, n);
 				}
			}
		}
		for(int i = 0;i < c5Layer.length;i++){
			double tempHidden = 0;
			for(int j = 0;j < gapLine.length;j++){
				tempHidden += gapLine[j] * gap5IH[i].getWeight(j);
			}
			c5Layer[i].setNumber(0, 0, MathFunction.sigmoid(tempHidden));
		}
		//Calculate NO.6 layer: f6 layer
		for(int i = 0;i < f6Layer.length;i++){
			double tempHidden = 0;
			for(int j = 0;j < c5Layer.length;j++){
				tempHidden += bp6HH[i].getWeight(j) * c5Layer[j].getNumber(0, 0);
			}
			f6Layer[i].setNumber(0, 0, MathFunction.sigmoid(tempHidden)); 
		}
		//Calculate NO.7 layer: output layer   
		for(int i = 0;i < outputLayer.length;i++){
			double tempOutput = 0;
			for(int j = 0;j < f6Layer.length;j++){
				tempOutput += bp7HO[i].getWeight(j) * f6Layer[j].getNumber(0, 0);
			}
			outputLayer[i].setNumber(0, 0, MathFunction.sigmoid(tempOutput));  
			//System.out.println(outputLayer[i].getNumber(0, 0));
		}
	}
	
	//To guess and save result.
	private void guessNumberAndSaveAnswer() throws IOException{
		double max = 0;
		for(int i = 0;i < outputLayer.length;i++){
			if(outputLayer[i].getNumber(0, 0) > max){
				max = outputLayer[i].getNumber(0, 0);
				guessNumber = i;
			}
		}
		File file = new File(resultPath);
		BufferedWriter fw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, true), "UTF-8"));
		fw.append("" + guessNumber);
		fw.newLine();
		fw.flush();
		fw.close();
	}
	
	private void backPropagation(){
		calculateSensitivation();
		adjustParameter();
	}
	
	private void calculateSensitivation(){
		//calculateOutputError(); and reset weight
		for(int i = 0; i < f6Layer.length; i++){
			for(int j = 0; j < outputLayer.length; j++){
				double change = 0;
				change = f6Layer[i].getNumber(0, 0) * outputLayer[j].getNumber(0, 0) 
						* (1 - outputLayer[j].getNumber(0, 0)) * (desiredOutput[j] - outputLayer[j].getNumber(0, 0));
				double temp = bp7HO[j].getWeight(i) + change;
				bp7HO[j].setWeight(i,temp);
			}
		}
		//calculateHidden2Error() and reset weight
		for(int i = 0; i < f6Layer.length; i++){	
			double sum = 0;
			for(int j = 0; j < outputLayer.length; j++){
				sum += bp7HO[j].getWeight(i) * outputLayer[j].getNumber(0, 0)
						* (1 - outputLayer[j].getNumber(0, 0)) * (desiredOutput[j] - outputLayer[j].getNumber(0, 0));
			}
			f6Layer[i].setError(0, 0, sum);
		}
		for(int i = 0; i < c5Layer.length; i++){
			for(int j = 0; j< f6Layer.length; j++){
				double change = 0;
				change = c5Layer[i].getNumber(0, 0) * f6Layer[j].getNumber(0, 0) 
						* (1 - f6Layer[j].getNumber(0, 0)) * f6Layer[j].getError(0, 0);
				double temp = bp6HH[j].getWeight(i) + change;
				bp6HH[j].setWeight(i, temp);
			}
		}
		
		//calculateHidden1Delta(); and reset 
		for(int i = 0; i < c5Layer.length; i++){
			double temp = 0;
			for(int j = 0; j < f6Layer.length; j++){
				temp += bp6HH[j].getWeight(i) * f6Layer[j].getError(0, 0) * f6Layer[j].getNumber(0, 0) 
						* (1 - f6Layer[j].getNumber(0, 0));
			}
			c5Layer[i].setError(0, 0, temp);
		}
		for(int i = 0; i < gapLine.length; i++){
			for(int j = 0; j < c5Layer.length; j++){
				double temp = 0;
				temp = c5Layer[j].getError(0, 0) * gapLine[i] * 
						c5Layer[j].getNumber(0, 0) * (1 - c5Layer[j].getNumber(0, 0));
			}
		}
		//calculateInputDelta();
		for(int i = 0; i < gapLine.length; i++){
			double temp = 0;
			for(int j = 0; j < c5Layer.length; j++){
				temp += c5Layer[j].getError(0, 0) * gap5IH[j].getWeight(i) * c5Layer[j].getNumber(0, 0)
						* (1 - c5Layer[j].getNumber(0, 0));
				gapError[i] = temp;
			}
		}
		//calculateS4Delta();
		int nn = 0;
		for(int i = 0; i < 16; i++){
			for(int j = 0; j < 5; j++){
				for(int k = 0; k < 5; k++){
					s4Layer[i].setError(j, k, gapError[nn]);
					nn++;
				}
			}
		}
		//calculateC3Delta();
		double[][][] temp = new double[16][10][10];
		for(int i = 0; i < 16; i++){
			temp[i] = MathFunction.upSampling(s4Layer[i].getErrorMatrix());
		}
		for(int i = 0; i < 16; i++){
			for(int j = 0; j < 10; j++){
				for(int k = 0; k < 10; k++){
					c3Layer[i].setError(j, k, temp[i][j][k] * c3Layer[i].getNumber(j, k));
				}
			}
		}
		//calculateS2Delta();
		for(int j = 0; j < 16; j++){
			for(int m = 0; m < 10; m++){
				for(int n = 0; n < 10; n++){
					for(int i = 0; i < 6; i++){
						for(int u = 0; u < 5; u++){
							for(int v = 0; v < 5; v++){
								double hereTemp = s2Layer[i].getError(u+m, v+n) + c3Layer[j].getError(m, n) *
										ck3Set[j][i].weight[u][v] * c3Layer[j].getNumber(m, n); 
								s2Layer[i].setError(u+m, v+n, hereTemp);
							}
						}
					}
				}
			}
		}
		double[][][][] c2Kernel = new double[6][16][5][5];
		double[][][] c1Kernel = new double[6][5][5];
		//calculateS2Error();
		for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 6; j++) {
				for (int m = 0; m < 10; m++) {
					for (int n = 0; n < 10; n++) {
						for (int p = 0; p < 5; p++) {
							for (int q = 0; q < 5; q++) {
								c2Kernel[j][i][p][q] += c3Layer[i].getError(m, n)
										* s2Layer[j].getNumber(m+p, n+q)
										* c3Layer[i].getNumber(m, n)
										* (1 - c3Layer[i].getNumber(m, n));
							}
						}
					}
				}
			}
		}
		//calculateC1Delta();
		temp = new double[6][28][28];
		for(int i = 0; i < 6; i++){
			temp[i] = MathFunction.upSampling(s2Layer[i].getErrorMatrix());
		}
		for(int i = 0; i < 6; i++){
			for(int j = 0; j < 28; j++){
				for(int k = 0; k < 28; k++){
					c1Layer[i].setError(j, k, temp[i][j][k] * c1Layer[i].getNumber(j, k));;
				}
			}
		}
		//calculateInputError();
		for(int i = 0; i < 6; i++){
			for(int m = 0; m < 28; m++){
				for(int n = 0; n < 28; n++){
					for(int j = 0; j < 5; j++){
						for(int k = 0; k < 5; k++){
							c1Kernel[i][j][k] += c1Layer[i].getError(m, n) * inputLayer[0].getNumber(m+j, n+k)
									* c1Layer[i].getNumber(m, n) * (1 - c1Layer[i].getNumber(m, n));
						}
					}
				}
			}
		}
		
	}
	
	private void adjustParameter(){
	
	}
	
	
	//Clean up all the parameter and saved output
	private void reset(){
		for(int i = 0;i < inputLayer.length;i++){
			inputLayer[i].reset();
		}
		for(int i = 0;i < c1Layer.length;i++){
			c1Layer[i].reset();
		}
		for(int i = 0;i < s2Layer.length;i++){
			s2Layer[i].reset();
		}
		for(int i = 0;i < c3Layer.length;i++){
			c3Layer[i].reset();
		}
		for(int i = 0;i < s4Layer.length;i++){
			s4Layer[i].reset();
		}
		for(int i = 0;i < c5Layer.length;i++){
			c5Layer[i].reset();
		}
		for(int i = 0;i < f6Layer.length;i++){
			f6Layer[i].reset();
		}
		for(int i = 0;i < outputLayer.length;i++){
			outputLayer[i].reset();
		}
	}
	

	//To read all the parameter on the disk
	private void readParaFromDisk() throws FileNotFoundException, ClassNotFoundException, IOException{
		for(int i = 0;i < ck1.length;i++){
			ck1[i].readFromDisk(weightSavePath);
		}
//		for(int i = 0; i < sk2.length;i++){
//			sk2[i].readFromDisk(weightSavePath);
//		}
		for(int i = 0;i < ck3Set.length;i++){
			for(int j = 0;j <ck3Set[i].length;j++){
				ck3Set[i][j].readFromDisk(weightSavePath);
			}
		}
//		for(int i = 0;i < sk4.length;i++){
//			sk4[i].readFromDisk(weightSavePath);
//		}
		for(int i = 0;i < gap5IH.length;i++){
			gap5IH[i].readFromDisk(weightSavePath);
		}
		for(int i = 0;i < bp6HH.length;i++){
			bp6HH[i].readFromDisk(weightSavePath);
		}
		for(int i = 0;i < bp7HO.length;i++){
			bp7HO[i].readFromDisk(weightSavePath);
		}
	}
	
	//To save all the parameters to the disk
	private void saveParaToDisk() throws FileNotFoundException, IOException{
		for(int i = 0;i < ck1.length;i++){
			ck1[i].saveToDisk(weightSavePath);
		}
//		for(int i = 0; i < sk2.length;i++){
//			sk2[i].saveToDisk(weightSavePath);
//		}
		for(int i = 0;i < ck3Set.length;i++){
			for(int j = 0;j < ck3Set[i].length;j++){
				ck3Set[i][j].saveToDisk(weightSavePath);
			}
		}
//		for(int i = 0;i < sk4.length;i++){
//			sk4[i].saveToDisk(weightSavePath);
//		}
		for(int i = 0;i < gap5IH.length;i++){
			gap5IH[i].saveToDisk(weightSavePath);
		}
		for(int i = 0;i < bp6HH.length;i++){
			bp6HH[i].saveToDisk(weightSavePath);
		}
		for(int i = 0;i < bp7HO.length;i++){
			bp7HO[i].saveToDisk(weightSavePath);
		}
	}
	
}
