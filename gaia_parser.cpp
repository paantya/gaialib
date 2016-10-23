#include <stdio.h>
#include <string.h>

int main (int argc, char* argv[]){
	
	FILE *file; 
	char *fname = "GaiaSource_000-000-000.csv";
	char result_sting[100000]; //Строка в 20 символов

	file = fopen(fname,"r");

	if(file == NULL)
	{
		printf("не могу открыть файл '%s'",fname);
		return 0;
	}
	int i=0;
	char *real_tail;
	int max = 0;

	while(fgets(result_sting,sizeof(result_sting),file))
	{
		real_tail = "";
		if (strlen(result_sting) > max){
			max = strlen(result_sting);
			printf("Строка %d:Длина строки - %d (max = %d):",i++,strlen(result_sting),max);
			printf("%s%s\n",result_sting,real_tail);
		}
		//printf("Строка %d:Длина строки - %d (max = %d):",i++,strlen(result_sting),max);

		if(result_sting[strlen(result_sting)-1] == '\n')//проверяем является ли последний элемент в строке символом её окончания
		{
			real_tail = "\\n";
			result_sting[strlen(result_sting)-1]='\0';
		};// эта часть кода добавлена лишь для отображения символа конца строки в консоль без перевода на новую строку	
		//printf("%s%s\n",result_sting,real_tail);
	}

	fclose(file);
return 0;
}