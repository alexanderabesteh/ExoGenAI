# Introduction
ExoGen AI is a full-stack platform that combines physics-based validation with deep learning to analyze stellar light curves from NASA's Kepler mission and identify potential exoplanets.

The system integrates a FastAPI backend for model inference and training with a Next.js frontend for data upload, visualization, and result interaction.

# ExoGen AI Backend Setup

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/alexanderabesteh/ExoGenAI.git
   cd ExoGenAI/backend
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download required files**

   - **Trained model**:  
     Download `best_model.pt` from https://drive.google.com/drive/folders/1O8w9Ls2u_D0gkZ8Trnpj9tayAbRcH3Gc?usp=sharing and place it in:
     ```
     backend/checkpoints/best_model.pt
     ```
   - **Dataset**:  
     Download from https://drive.google.com/drive/folders/1O8w9Ls2u_D0gkZ8Trnpj9tayAbRcH3Gc?usp=sharing.(labelled time-series data)
     Place these files:
     ```
     backend/data/exoTrain.csv
     backend/data/exoTest.csv
     ```

4. **Start API server**

   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

5. **Test the API**

   - Health check:  
     [http://localhost:8000/health](http://localhost:8000/health)
   - Documentation (Swagger):  
     [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoint

### **POST /predict-with-plot**

- **Upload:** CSV file with `time` and `flux` columns.
- **Returns:**  
  - Prediction  
  - Probabilities  
  - plot_data  

## Troubleshooting

- **No GPU detected?**  
  Edit `api.py` line 38:  
  Change `device='cuda'` to `device='cpu'`.

- **Port conflict?**  
  Start the server with a different port:
  ```bash
  uvicorn api:app --host 0.0.0.0 --port 8001
  ```

***

# ExoGen AI Frontend Setup

## Quick Start
1. **Navigate to the frontend directory**

   ```bash
   cd ../frontend
   ```

2. **Install dependencies**

   ```bash
   npm install
    ```
3. **Start the development server**
    ```bash
    npm run dev
    ```

4. **Open your browser and navigate to**
    [http://localhost:3000](http://localhost:3000)
5. **Upload a CSV file and view results**

# ExoGen AI Full Stack

## Quick Start
Ensure both backend and frontend are set up as per the instructions above.

1. **Start the backend server**
    ```bash
    cd ../backend
    uvicorn api:app --host 0.0.0.0 --port 8000
    ```
2. In a new terminal, **start the frontend server**
    ```bash
    cd ../frontend
    npm run dev
3. **Open your browser and navigate to**
    [http://localhost:3000](http://localhost:3000)
4. **Upload a CSV file and view results**
