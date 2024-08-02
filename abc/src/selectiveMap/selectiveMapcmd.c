#include "base/main/main.h"
#include "selectiveMap.h"

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

static int Abc_CommandSelectiveMap(Abc_Frame_t* pAbc, int argc, char** argv);


////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////


void selectiveMap_Init( Abc_Frame_t * pAbc )
{
    Cmd_CommandAdd( pAbc, "SC mapping", "selective_map", Abc_CommandSelectiveMap, 0);
}


int Abc_CommandSelectiveMap( Abc_Frame_t * pAbc, int argc, char ** argv ){
    Abc_Ntk_t * pNtk, * pNtkRes;
    char Buffer[100];
    double DelayTarget;
    double AreaMulti;
    double DelayMulti;
    float LogFan = 0;
    float Slew = 0; // choose based on the library
    float Gain = 250;
    int nGatesMin = 0;
    int fAreaOnly;
    int fRecovery;
    int fSweep;
    int fSwitching;
    int fSkipFanout;
    int fUseProfile;
    int fUseBuffs;
    int fVerbose;
    int c;

    extern int Abc_NtkFraigSweep( Abc_Ntk_t * pNtk, int fUseInv, int fExdc, int fVerbose, int fVeryVerbose );

    pNtk = Abc_FrameReadNtk(pAbc);
    // set defaults
    DelayTarget =-1;
    AreaMulti   = 0;
    DelayMulti  = 0;
    fAreaOnly   = 0;
    fRecovery   = 1;
    fSweep      = 0;
    fSwitching  = 0;
    fSkipFanout = 0;
    fUseProfile = 0;
    fUseBuffs   = 0;
    fVerbose    = 0;

    char* cutChoiceFile = "";
    char* dumpFile = "";

    int cutChoice_flag = 0; 
    int dumpFile_flag = 0;

    Extra_UtilGetoptReset();

    while ( ( c = Extra_UtilGetopt( argc, argv, "DABFSGMarspfuovhcm" ) ) != EOF )
    {
        switch ( c )
        {
        case 'D':
            if ( globalUtilOptind >= argc )
            {
                Abc_Print( -1, "Command line switch \"-D\" should be followed by a floating point number.\n" );
                goto usage;
            }
            DelayTarget = (float)atof(argv[globalUtilOptind]);
            globalUtilOptind++;
            if ( DelayTarget <= 0.0 )
                goto usage;
            break;
        case 'A':
            if ( globalUtilOptind >= argc )
            {
                Abc_Print( -1, "Command line switch \"-A\" should be followed by a floating point number.\n" );
                goto usage;
            }
            AreaMulti = (float)atof(argv[globalUtilOptind]);
            globalUtilOptind++;
            break;
        case 'B':
            if ( globalUtilOptind >= argc )
            {
                Abc_Print( -1, "Command line switch \"-B\" should be followed by a floating point number.\n" );
                goto usage;
            }
            DelayMulti = (float)atof(argv[globalUtilOptind]);
            globalUtilOptind++;
            break;
        case 'F':
            if ( globalUtilOptind >= argc )
            {
                Abc_Print( -1, "Command line switch \"-F\" should be followed by a floating point number.\n" );
                goto usage;
            }
            LogFan = (float)atof(argv[globalUtilOptind]);
            globalUtilOptind++;
            if ( LogFan < 0.0 )
                goto usage;
            break;
        case 'S':
            if ( globalUtilOptind >= argc )
            {
                Abc_Print( -1, "Command line switch \"-S\" should be followed by a floating point number.\n" );
                goto usage;
            }
            Slew = (float)atof(argv[globalUtilOptind]);
            globalUtilOptind++;
            if ( Slew <= 0.0 )
                goto usage;
            break;
        case 'G':
            if ( globalUtilOptind >= argc )
            {
                Abc_Print( -1, "Command line switch \"-G\" should be followed by a floating point number.\n" );
                goto usage;
            }
            Gain = (float)atof(argv[globalUtilOptind]);
            globalUtilOptind++;
            if ( Gain <= 0.0 )
                goto usage;
            break;
        case 'M':
            if ( globalUtilOptind >= argc )
            {
                Abc_Print( -1, "Command line switch \"-M\" should be followed by a positive integer.\n" );
                goto usage;
            }
            nGatesMin = atoi(argv[globalUtilOptind]);
            globalUtilOptind++;
            if ( nGatesMin < 0 )
                goto usage;
            break;
        case 'c':
            if ( globalUtilOptind >= argc )
            {
                goto usage;
            }
            cutChoiceFile = argv[globalUtilOptind];
            globalUtilOptind++;
            cutChoice_flag = 1;
            break;
        case 'm':
            if ( globalUtilOptind >= argc )
            {
                goto usage;
            }
            dumpFile = argv[globalUtilOptind];
            globalUtilOptind++;
            dumpFile_flag = 1;
            break;
        case 'a':
            fAreaOnly ^= 1;
            break;
        case 'r':
            fRecovery ^= 1;
            break;
        case 's':
            fSweep ^= 1;
            break;
        case 'p':
            fSwitching ^= 1;
            break;
        case 'f':
            fSkipFanout ^= 1;
            break;
        case 'u':
            fUseProfile ^= 1;
            break;
        case 'o':
            fUseBuffs ^= 1;
            break;
        case 'v':
            fVerbose ^= 1;
            break;
        case 'h':
            goto usage;
        default:
            goto usage;
        }
    }

    if (pNtk == NULL)
    {
        Abc_Print( -1, "Empty network.\n" );
        return 1;
    }
    
    if ( fAreaOnly )
        DelayTarget = ABC_INFINITY;

    if(!cutChoice_flag)
    {
        Abc_Print(-1, "Need a cut choice file to read the cuts\n");
        return 1;
    }
    else if(!dumpFile_flag){
        Abc_Print(-1, "Need a dump file to dump the cuts used in the final mapping\n");
        return 1;
    }
    else if(!Abc_NtkIsStrash(pNtk)){
        pNtk = Abc_NtkStrash( pNtk, 0, 0, 0 );
        if ( pNtk == NULL )
        {
            Abc_Print( -1, "Strashing before mapping has failed.\n" );
            return 1;
        }
        pNtk = Abc_NtkBalance( pNtkRes = pNtk, 0, 0, 1 );
        Abc_NtkDelete( pNtkRes );
        if ( pNtk == NULL )
        {
            Abc_Print( -1, "Balancing before mapping has failed.\n" );
            return 1;
        }
        Abc_Print( 0, "The network was strashed and balanced before mapping.\n" );
        pNtkRes = Abc_selectiveMap(pNtk, cutChoiceFile, dumpFile, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin, fRecovery, fSwitching, fSkipFanout, fUseProfile, fUseBuffs, fVerbose);
        if ( pNtkRes == NULL )
        {
            Abc_NtkDelete( pNtk );
            Abc_Print( -1, "Mapping has failed.\n" );
            return 1;
        }
        Abc_NtkDelete( pNtk );
    }else{
        pNtkRes = Abc_selectiveMap(pNtk, cutChoiceFile, dumpFile, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin, fRecovery, fSwitching, fSkipFanout, fUseProfile, fUseBuffs, fVerbose);
        if ( pNtkRes == NULL )
        {
            Abc_Print( -1, "Mapping has failed.\n" );
            return 1;
        }
    }

    if ( fSweep )
    {
        Abc_NtkFraigSweep( pNtkRes, 0, 0, 0, 0 );
        if ( Abc_NtkHasMapping(pNtkRes) )
        {
            pNtkRes = Abc_NtkDupDfs( pNtk = pNtkRes );
            Abc_NtkDelete( pNtk );
        }
    }

    // replace the current network
    Abc_FrameReplaceCurrentNetwork( pAbc, pNtkRes );
    return 0;

    usage:
        if ( DelayTarget == -1 )
            sprintf(Buffer, "not used" );
        else
            sprintf(Buffer, "%.3f", DelayTarget );
        Abc_Print( -2, "usage: map [-DABFSG float] [-M num] [-arspfuovh]\n" );
        Abc_Print( -2, "\t           performs standard cell mapping of the current network\n" );
        Abc_Print( -2, "\t-D float : sets the global required times [default = %s]\n", Buffer );
        Abc_Print( -2, "\t-A float : \"area multiplier\" to bias gate selection [default = %.2f]\n", AreaMulti );
        Abc_Print( -2, "\t-B float : \"delay multiplier\" to bias gate selection [default = %.2f]\n", DelayMulti );
        Abc_Print( -2, "\t-F float : the logarithmic fanout delay parameter [default = %.2f]\n", LogFan );
        Abc_Print( -2, "\t-S float : the slew parameter used to generate the library [default = %.2f]\n", Slew );
        Abc_Print( -2, "\t-G float : the gain parameter used to generate the library [default = %.2f]\n", Gain );
        Abc_Print( -2, "\t-M num   : skip gate classes whose size is less than this [default = %d]\n", nGatesMin );
        Abc_Print( -2, "\t-a       : toggles area-only mapping [default = %s]\n", fAreaOnly? "yes": "no" );
        Abc_Print( -2, "\t-r       : toggles area recovery [default = %s]\n", fRecovery? "yes": "no" );
        Abc_Print( -2, "\t-s       : toggles sweep after mapping [default = %s]\n", fSweep? "yes": "no" );
        Abc_Print( -2, "\t-p       : optimizes power by minimizing switching [default = %s]\n", fSwitching? "yes": "no" );
        Abc_Print( -2, "\t-f       : do not use large gates to map high-fanout nodes [default = %s]\n", fSkipFanout? "yes": "no" );
        Abc_Print( -2, "\t-u       : use standard-cell profile [default = %s]\n", fUseProfile? "yes": "no" );
        Abc_Print( -2, "\t-o       : toggles using buffers to decouple combinational outputs [default = %s]\n", fUseBuffs? "yes": "no" );
        Abc_Print( -2, "\t-v       : toggles verbose output [default = %s]\n", fVerbose? "yes": "no" );
        Abc_Print( -2, "\t-h       : print the command usage\n");
        Abc_Print( -2, "\t-c <cutChoice_file.csv> -m <mapFile.csv>       : Reads the cut choices, does selective mapping and dumps the cuts used in it\n");
        return 1;
}  

ABC_NAMESPACE_IMPL_END