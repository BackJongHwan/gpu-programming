# GPU Programming Assignments

이 저장소는 GPU 프로그래밍 수업의 과제들을 포함하고 있습니다. CUDA를 사용하여 다양한 연산을 가속화하고 성능을 최적화하는 방법을 다룹니다.

## 과제 목록

### HW 1: 단위 구의 부피 계산 (Unit Sphere Volume)
- **파일**: [HW_1_20201587.cu](file:///c:/Users/wantg/Downloads/HW_2_20201587/HW_1_20201587.cu)
- **설명**: 몬테카를로 시뮬레이션(Monte Carlo Simulation)과 병렬 리덕션(Parallel Reduction)을 사용하여 3차원 단위 구의 부피를 계산합니다.
- **주요 기술**:
    - CUDA 커널 작성
    - Parallel Reduction 최적화
    - Thrust 라이브러리 활용

### HW 2: 행렬 곱셈 최적화 (Matrix Multiplication Optimization)
- **파일**: [HW_2_20201587.cu](file:///c:/Users/wantg/Downloads/HW_2_20201587/HW_2_20201587.cu)
- **설명**: 다양한 기법을 사용하여 행렬 곱셈(GEMM) 연산을 수행하고 성능을 비교합니다.
- **구현된 방법**:
    1. **Global Memory (GM)**: Shared Memory 없이 구현
    2. **Shared Memory (SM)**: Tiling 기법 적용
    3. **SM + More-work-per-thread (WPT)**: 스레드당 더 많은 작업 할당 및 레지스터 활용
    4. **Tensor Core (GM)**: Tensor Core를 사용한 연산 (Global Memory 기반)
    5. **Tensor Core (SM)**: Shared Memory와 Tensor Core를 조합한 최적화
    6. **cuBLAS (CUDA Core)**: cuBLAS 라이브러리 사용
    7. **cuBLAS (Tensor Core)**: cuBLAS의 Tensor Core 가속 모드 사용

## 사용 방법

### 컴파일 환경
- NVIDIA GPU (Compute Capability 7.0 이상 권장 - Tensor Core 과제 수행 시)
- CUDA Toolkit 설치됨
- `nvcc` 컴파일러

### 컴파일 및 실행 예시 (HW 2)
```bash
nvcc -o HW2 HW_2_20201587.cu -lcublas
./HW2
```

## 제출 리포트
- [HW 1 리포트 (docx)](file:///c:/Users/wantg/Downloads/HW_2_20201587/HW_1_20201587.docx)
- [HW 2 리포트 (pdf)](file:///c:/Users/wantg/Downloads/HW_2_20201587/HW_2_20201587.pdf)
