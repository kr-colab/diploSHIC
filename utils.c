#include <math.h>

double r2(int nSamps, int *haps, int i, int j){
	double pi  = 0.0;
	double pj  = 0.0;
	double pij = 0.0;
	double count = 0.0;
	int k;
	for(k=0; k<nSamps; k++){
		if((haps[i*nSamps + k] == 1 || haps[i*nSamps + k] == 0) && (haps[j*nSamps + k] != 1 || haps[j*nSamps + k] != 0)){
			if(haps[i*nSamps + k] == 1)
				pi++;

			if(haps[j*nSamps + k] == 1)
				pj++;

			if(haps[i*nSamps + k] == 1 && haps[j*nSamps + k] == 1)
				pij++;
			count += 1;
		}
	}
	if (count == 0){
		return(-1.0);
	}
	else{
		pi  /= count;
		pj  /= count;
		pij /= count;

		double Dij = pij - (pi*pj);

		return (Dij*Dij) / ((pi*(1.0-pi)) * (pj*(1.0-pj)));
	}
}

void computeR2Matrix(int nSamps, int nSnps, int *haps, double *r2Matrix){
	double r2Val;
	int i, j;
	for (i=0; i<nSnps-1; i++){
		for (j=i+1; j<nSnps; j++){
			r2Val = r2(nSamps, haps, i, j);
			r2Matrix[i*nSnps +j] = r2Val;
		}
	}
}

void ZnS(int nSnps, double *r2Matrix, double *zns){
	double r2Sum = 0;
	int r2Denom = 0;
	int i, j;
	if (nSnps < 2)
		*zns = 0.0;
	else{
		for (i=0; i<nSnps-1; i++){
			for (j=i+1; j<nSnps; j++){
				if (r2Matrix[i*nSnps + j] >= 0.0){
					r2Sum += r2Matrix[i*nSnps + j];
					r2Denom += 1;
				}
			}
		}
		*zns = r2Sum/(float)r2Denom;
	}
}

double omegaAtSnp(int l, int nSnps, double *r2Matrix){
	double oSum=0.0, oSumL=0.0, oSumR = 0.0;
	double denom, numer;
	int lCount=0, rCount=0, crossCount=0;
	int i, j;

	for (i=0; i<nSnps-1; i++){
		for (j=i+1; j<nSnps; j++){
			if (r2Matrix[i*nSnps + j] >= 0.0){
				if (i < l && j >= l){
					oSum += r2Matrix[i*nSnps + j];
					crossCount += 1;
				}
				else if (i < l && j < l){
					oSumL += r2Matrix[i*nSnps + j];
					lCount += 1;
				}
				else if (i >= l && j >= l){
					oSumR += r2Matrix[i*nSnps + j];
					rCount += 1;
				}
			}
		}
	}
	denom = oSum * (1.0/(float) crossCount);
	numer = 1.0 / (float) (lCount + rCount);
	numer *= (oSumL+oSumR);
	return numer/denom;
}

void omega(int nSnps, double *r2Matrix, double *omegaMax){
	*omegaMax = 0.0;
	double tmp;
	int l;

	if (nSnps < 3)
		*omegaMax = 0.0;
	else{
		for (l=3; l<nSnps-2; l++){
			tmp = omegaAtSnp(l, nSnps, r2Matrix);
			if (tmp > *omegaMax)
				*omegaMax = tmp;
		}
	}
}

void pairwiseDiffs(int nSamps, int nSnps, int *haps, double *diffLs){
	int i, j, k, basei, basej, diffs;
	int pairsSeen = 0;
	for(i=0; i<nSamps-1; i++){
		for(j=i+1; j<nSamps; j++){
			diffs = 0;
			for(k=0; k<nSnps; k++){
				basei = haps[k*nSamps + i];
				basej = haps[k*nSamps + j];
				if(basei >= 0 && basei <= 1 && basej >= 0 && basej <= 1){
					if (basei != basej){
						diffs += 1;
					}
				}
			}
			diffLs[pairsSeen] = diffs;
			pairsSeen += 1;
		}
	}
}

void pairwiseDiffsDiplo(int nSamps, int nSnps, int *haps, double *diffLs){
	int i, j, k, basei, basej, diffs;
	int pairsSeen = 0;
	for(i=0; i<nSamps-1; i++){
		for(j=i+1; j<nSamps; j++){
			diffs = 0;
			for(k=0; k<nSnps; k++){
				basei = haps[k*nSamps + i];
				basej = haps[k*nSamps + j];
				if(basei >= 0 && basei <= 2 && basej >= 0 && basej <= 2){
					if (basei != basej){
						diffs += 1;
					}
				}
			}
			diffLs[pairsSeen] = diffs;
			pairsSeen += 1;
		}
	}
}
