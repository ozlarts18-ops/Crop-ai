import sys
import time
from src.stage1_problem import run_stage1
from src.stage2_collect import run_stage2
from src.stage3_clean import run_stage3
from src.stage4_features import run_stage4
from src.stage5_split import run_stage5
from src.stage6_train_cnn import train_cnn
from src.stage6_train_yolo import run_yolo_stage
from src.stage7_evaluate import run_stage7

def main():
    print("*"*80)
    print(" CROP_AI: SOYBEAN CROP HEALTH & DISEASE DETECTION PIPELINE (STAGES 1 TO 7)")
    print("*"*80)
    start_time = time.time()
    
    # Stage 1: Real-World Problem
    run_stage1()
    
    # Stage 2: Collect Data
    run_stage2()
    
    # Stage 3: Clean / Prepare Data
    run_stage3()
    
    # Stage 4: Identify Features & Target
    run_stage4()
    
    # Stage 5: Split Data
    run_stage5()
    
    # Stage 6: Train Models
    print("\n" + "="*80)
    print(" STAGE 6: MODEL TRAINING")
    print("="*80)
    yolo_status = run_yolo_stage()
    best_cnn_model = train_cnn(epochs=100, batch_size=32)
    
    # Stage 7: Evaluate Model
    print("\n" + "="*80)
    print(" STAGE 7: MODEL EVALUATION & TESTING")
    print("="*80)
    final_results = run_stage7()
    
    total_time = time.time() - start_time
    print(f"\nAll pipeline stages (1-7) executed successfully in {total_time:.1f}s ({total_time/60:.2f} mins).")

if __name__ == "__main__":
    main()
