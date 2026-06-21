#include <stdlib.h>
#include <stdio.h>
#include <string.h>

double *read_double_column(FILE *csv, int column_number, int *lenght){
    char line[1024];
    double *column = NULL;
    const char *delimeter =",";
    char *tok;
    char *end;

    if (fgets(line, sizeof(line), csv) == NULL) {
        fclose(csv);
        exit(1);
    }

    int j=0;
    while (fgets(line, sizeof(line), csv)){
        int i=0;
        tok = strtok(line, delimeter);

        while(tok !=NULL && i<column_number){
            tok = strtok(NULL, delimeter);
            i++;
        }

        if(tok == NULL){
            continue;
        }

        double db = strtod(tok ,&end);
        if(tok == end){
            continue;
        }
        
        column = realloc(column,sizeof(double) * (j+1));

        if (column == NULL) {
            printf("Memory reallocation failed.\n");
            free(column);
            exit(1);
        }

        column[j++]=db;
        *lenght = j;
    }
    return column;
}

char **read_string_column(FILE *csv, int column_number, int *lenght){
    char line[1024];
    char **column = NULL;
    const char *delimeter =",";
    char *tok;

    if (fgets(line, sizeof(line), csv) == NULL) {
        fclose(csv);
        exit(1);
    }

    int j=0;
    while (fgets(line, sizeof(line), csv)){
        int i=0;
        tok = strtok(line, delimeter);

        while(tok !=NULL && i<column_number){
            tok = strtok(NULL, delimeter);
            i++;
        }

        if(tok == NULL){
            continue;
        }

        size_t len = strlen(tok);
        while (len > 0 && (tok[len-1] == '\n' || tok[len-1] == '\r')) {
            tok[--len] = '\0';
        }

        
        char **temp = realloc(column,sizeof(char *) * (j+1));

        if (temp == NULL) {
            printf("Memory reallocation failed.\n");
            for (int k = 0; k < j; k++) free(column[k]);
            free(column);
            exit(1);
        }
        column = temp;

        column[j] = strdup(tok);
        if (column[j] == NULL) {
            printf("strdup failed.\n");
            for (int k = 0; k < j; k++) free(column[k]);
            free(column);
            exit(1);
        }
        j++;
        
    }
    *lenght = j;
    return column;
}