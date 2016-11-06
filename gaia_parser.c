#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct gaiafold
{
	long solution_id;
	long source_id;
	long random_index;
	double ref_epoch;
	double ra;
	double ra_error;
	double dec;
	double dec_error;
	double parallax;
	double parallax_error;
	double pmra;
	double pmra_error;
	double pmdec;
	double pmdec_error;
	float ra_dec_corr;
	float ra_parallax_corr;
	float ra_pmra_corr;
	float ra_pmdec_corr;
	float dec_parallax_corr;
	float dec_pmra_corr;
	float dec_pmdec_corr;
	float parallax_pmra_corr;
	float parallax_pmdec_corr;
	float pmra_pmdec_corr;
	int astrometric_n_obs_al;
	int astrometric_n_obs_ac;
	int astrometric_n_good_obs_al;
	int astrometric_n_good_obs_ac;
	int astrometric_n_bad_obs_al;
	int astrometric_n_bad_obs_ac;
	float astrometric_delta_q;
	double astrometric_excess_noise;
	double astrometric_excess_noise_sig;
	bool astrometric_primary_flag;
	float astrometric_relegation_factor;
	float astrometric_weight_al;
	float astrometric_weight_ac;
	int astrometric_priors_used;
	short matched_observations;
	bool duplicated_source;
	float scan_direction_strength_k1;
	float scan_direction_strength_k2;
	float scan_direction_strength_k3;
	float scan_direction_strength_k4;
	float scan_direction_mean_k1;
	float scan_direction_mean_k2;
	float scan_direction_mean_k3;
	float scan_direction_mean_k4;
	int phot_g_n_obs;
	double phot_g_mean_flux;
	double phot_g_mean_flux_error;
	double phot_g_mean_mag;
	double phot_variable_flag;
	double l;
	double b;
	double ecl_lon;
	double ecl_lat;
};


void splitcsv(char flag,char instr[1000])
{
	//printf("all ok, btgin\n");
	int ind=0,itmp=0;
	char strpars[100][100];
	if(instr[strlen(instr)-1] == '\n')
	//проверяем является ли последний элемент в строке символом её окончания
	{
		instr[strlen(instr)-1]='\0';
	}

	for (size_t i = 0; i <strlen(instr); ++i)
	{
		if (instr[i] == flag)
		{
			strncat(strpars[ind],&instr[i-itmp],itmp);		
			++ind;
			itmp = 0;
		}
		else
		{	
			++itmp;
		}

	}
	strncat(strpars[ind],&instr[strlen(instr)-itmp],itmp);		

	//обнуление, для возможности работы следующему циклу
	for (size_t i = 0; i <100; ++i)
	{
		strpars[i][0] = '\0';
	}
	//printf("all ok, end\n");
}


//atof()
int main (int argc, char* argv[]){
	
	FILE *file; 
	char *fname = "GaiaSource_000-000-000.csv";	
	file = fopen(fname,"r");

	if(file == NULL)
	{
		printf("Не могу открыть файл '%s'",fname);
		return 0;
	}
	
	int i=0;
	bool max = 0;
	char result_sting[1000]; //Строка в 1000символов

	while(fgets(result_sting, sizeof(result_sting), file))
	{
		if (max == 1)
		{			
			splitcsv(',',result_sting);
		}
		else 
		{
			max = 1;
			printf("%s\n",result_sting);
		}
	}
	fclose(file);
return 0;
}
