#include "base/main/main.h"
#include "getNCInfo.h"

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

static int Abc_CommandGetNCInfo(Abc_Frame_t* pAbc, int argc, char** argv);


////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////


void getNCInfo_Init( Abc_Frame_t * pAbc )
{
    Cmd_CommandAdd( pAbc, "SC mapping", "get_nc_info", Abc_CommandGetNCInfo, 0);
}


int Abc_CommandGetNCInfo( Abc_Frame_t * pAbc, int argc, char ** argv ){
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

    Abc_Ntk_t * pNtk = Abc_FrameReadNtk(pAbc);
    if (pNtk == NULL)
    {
        printf("There is no AIG\n");
        return 0;
    }

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

    int c = 0;

    char* cutsFile = "";
    char* nodesFile = "";

    int cuts_flag = 0; 
    int nodes_flag = 0; 

    Extra_UtilGetoptReset();

    while ( ( c = Extra_UtilGetopt( argc, argv, "hnc" ) ) != EOF )
    {
        switch ( c )
        {
            case 'h':
                goto usage;
            case 'n':
                if ( globalUtilOptind >= argc )
                {
                    goto usage;
                }
                nodesFile = argv[globalUtilOptind];
                globalUtilOptind++;
                nodes_flag = 1;
                break;
            case 'c':
                if ( globalUtilOptind >= argc )
                {
                    goto usage;
                }
                cutsFile = argv[globalUtilOptind];
                globalUtilOptind++;
                cuts_flag = 1;
                break;
            default:
                goto usage;
        }
    }
    
    if(!nodes_flag)
    {
        Abc_Print(-1, "Need an output file to dump the node features\n");
    }
    else if(!cuts_flag)
    {
        Abc_Print(-1, "Need an output file to dump the cut features\n");
    }
    else{
        if ( !Abc_NtkIsStrash(pNtk) )
        {
            pNtk = Abc_NtkStrash( pNtk, 0, 0, 0 );
            if ( pNtk == NULL )
            {
                Abc_Print( -1, "Strashing before preparing for map has failed.\n" );
                return 1;
            }
        }
        int res = Abc_getNCInfo(pNtk, nodesFile, cutsFile, DelayTarget, AreaMulti, DelayMulti, LogFan, Slew, Gain, nGatesMin, fRecovery, fSwitching, fSkipFanout, fUseProfile, fUseBuffs, fVerbose);
        if (res == 1) return 1;
        else return 0;
    }

    usage:
        Abc_Print( -2, "usage: get_nc_info -n <nodes_file.csv> -c <cuts_file.csv>: Generates and dumps the nodes and cuts features\n" );
        Abc_Print( -2, "usage: get_nc_info -h: Shows the usage features of the command\n" );
        return 0;
}  

ABC_NAMESPACE_IMPL_END