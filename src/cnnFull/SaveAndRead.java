package cnnFull;
import java.io.FileNotFoundException;
import java.io.IOException;

public interface SaveAndRead {
	public abstract void readFromDisk(String path) throws FileNotFoundException, IOException, ClassNotFoundException;
	public abstract void saveToDisk(String path) throws FileNotFoundException, IOException;
}
