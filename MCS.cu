#include <cstdio>
#include <cassert>
#include <chrono>

#include "Structure.cuh"

void CheckInput(SuperCell *self) {
    fprintf(stdout, "SuperCell Scale : %d %d %d\n", self->a, self->b, self->c);
    fprintf(stdout, "UnitCell a(%6.2f, %6.2lf, %6.2lf)\n", (self->unitCell->a).x, (self->unitCell->a).y, (self->unitCell->a).z);
    fprintf(stdout, "UnitCell b(%6.2f, %6.2lf, %6.2lf)\n", (self->unitCell->b).x, (self->unitCell->b).y, (self->unitCell->b).z);
    fprintf(stdout, "UnitCell c(%6.2f, %6.2lf, %6.2lf)\n", (self->unitCell->c).x, (self->unitCell->c).y, (self->unitCell->c).z);
    fprintf(stdout, "UnitCell %d dots.\n", self->unitCell->N);
    int N = self->unitCell->N;
    for (int i = 0; i < N; ++i) {
        fprintf(stdout, "Dot %d :\n", i);
        Dot* dot = self->unitCell->Dots; dot = dot + i;
        fprintf(stdout, "    Position (%6.2lf, %6.2lf, %6.2lf)\n", dot->Pos->x, dot->Pos->y, dot->Pos->z);
        fprintf(stdout, "    Value    (%6.2lf, %6.2lf, %6.2lf)\n", dot->a->x, dot->a->y, dot->a->z);
        fprintf(stdout, "    Ani      (%6.2lf, %6.2lf, %6.2lf,\n", dot->A->xx, dot->A->xy, dot->A->xz);
        fprintf(stdout, "              %6.2lf, %6.2lf, %6.2lf,\n", dot->A->yx, dot->A->yy, dot->A->yz);
        fprintf(stdout, "              %6.2lf, %6.2lf, %6.2lf)\n", dot->A->zx, dot->A->zy, dot->A->zz);
        Bond *bonds = (self->unitCell->Dots)[i].bonds;
        while (bonds != NULL) {
            fprintf(stdout, "        To %d, Overlat %d, %d, %d\n", bonds->t, bonds->Gx, bonds->Gy, bonds->Gz);
            fprintf(stdout, "        Corr (%6.2lf, %6.2lf, %6.2lf,\n", bonds->A->xx, bonds->A->xy, bonds->A->xz);
            fprintf(stdout, "              %6.2lf, %6.2lf, %6.2lf,\n", bonds->A->yx, bonds->A->yy, bonds->A->yz);
            fprintf(stdout, "              %6.2lf, %6.2lf, %6.2lf)\n", bonds->A->zx, bonds->A->zy, bonds->A->zz);
            bonds = bonds->Next;
        }
    }
    return;
}

int main() {
    SuperCell *superCell = NULL;
    FILE *structureInput = fopen("Input_Structure", "r");
    superCell = InitStructure(superCell, structureInput);
    if (superCell == NULL) {
        fprintf(stderr, "[ERROR] Failed loading structure. Exit.\n");
        return 0;
    }
    //CheckInput(superCell);
    DestroySuperCell(superCell);
    return 0;
}