NVCC ?= nvcc
TARGET := batch_grayscale
SRC := src/batch_grayscale.cu
NVCCFLAGS := -O2 -std=c++17

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET) $(TARGET).exe

generate:
	python3 scripts/generate_dataset.py --count 150

run: $(TARGET)
	./$(TARGET) --input images/input --output images/output --limit 150
