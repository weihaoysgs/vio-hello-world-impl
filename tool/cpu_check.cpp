#include <unistd.h>  // for sleep()

#include <cstdlib>  // for system()
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct CPUData
{
  long long user;
  long long nice;
  long long system;
  long long idle;
  long long iowait;
  long long irq;
  long long softirq;
  long long steal;
  long long guest;
  long long guest_nice;
};

void readStats( std::vector<CPUData>& entries ) {
  std::ifstream file( "/proc/stat" );
  std::string line;

  while ( std::getline( file, line ) ) {
    // Parse line if it starts with "cpu" but not "cpu "
    if ( line.substr( 0, 3 ) == "cpu" && line[3] != ' ' ) {
      CPUData data;
      sscanf( line.c_str(), "cpu%*d %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld", &data.user,
              &data.nice, &data.system, &data.idle, &data.iowait, &data.irq, &data.softirq,
              &data.steal, &data.guest, &data.guest_nice );
      entries.push_back( data );
    }
  }
  file.close();
}

double calculateCPUUsage( const CPUData& e1, const CPUData& e2 ) {
  const long long idle = e2.idle - e1.idle + e2.iowait - e1.iowait;
  const long long total = ( e2.user - e1.user ) + ( e2.nice - e1.nice ) +
                          ( e2.system - e1.system ) + idle + ( e2.irq - e1.irq ) +
                          ( e2.softirq - e1.softirq ) + ( e2.steal - e1.steal );
  return ( 1.0 - idle / (double)total ) * 100.0;
}

void printProgressBar( double percentage, bool average = false ) {
  if ( !average ) {
    int val = (int)( percentage * 100 );
    int lpad = (int)( percentage * 50 );
    int rpad = 50 - lpad;
    std::cout << "\033[1;32m";  // green color start
    std::cout << std::string( lpad, '|' );
    std::cout << "\033[0m";  // color reset
    std::cout << std::string( rpad, ' ' );
    std::cout << " - " << std::setw( 2 ) << val << "%";
  } else {
    int val = (int)( percentage * 100 );
    int lpad = (int)( percentage * 50 );
    int rpad = 50 - lpad;
    std::cout << std::string( lpad, '|' );
    std::cout << std::string( rpad, ' ' );
    std::cout << " - " << std::setw( 2 ) << val << "%";
  }
}
#define BLUE "\033[0;1;34m"
#define TAIL "\033[0m"
int main() {
  std::vector<CPUData> entries1, entries2;

  // Initial read
  readStats( entries1 );
  sleep( 1 );  // wait a second

  while ( true ) {
    // Clear previous output
    std::system( "clear" );

    // Read current stats
    entries2.clear();
    readStats( entries2 );

    std::cout << "CPU Usage by Core:" << std::endl;
    double average = 0;
    for ( size_t i = 0; i < entries1.size(); ++i ) {
      std::cout << std::left << std::setw( 4 ) << "CPU" << std::setw( 2 ) << i << ": ";
      double usage = calculateCPUUsage( entries1[i], entries2[i] );
      printProgressBar( usage / 100.0 );
      average += usage;
      std::cout << std::endl;
    }
    std::cout << BLUE << std::left << std::setw( 8 ) << "AVERAG:";
    printProgressBar( average / 100.0 / entries1.size(), true );
    std::cout << TAIL << std::endl;
    // Prepare for the next read
    entries1 = entries2;
    sleep( 1 );  // wait a second
  }

  return 0;
}
#undef BLUE
#undef TAIL